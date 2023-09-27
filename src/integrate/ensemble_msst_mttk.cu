/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
This integrator use Nosé-Hoover thermostat and Parrinello-Rahman barostat.
only P is set -> NPH ensemable
only T is set -> NVT ensemable
P and T are both set -> NPT ensemable
------------------------------------------------------------------------------*/

#include "ensemble_msst_mttk.cuh"

namespace
{

__device__ void matrix_vector_multiply(double a[3][3], double b[3], double c[3])
{
  for (int i = 0; i < 3; i++) {
    c[i] = 0;
    for (int j = 0; j < 3; j++)
      c[i] += a[i][j] * b[j];
  }
}

} // namespace

Ensemble_MSST_MTTK::Ensemble_MSST_MTTK(const char** params, int num_params)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      h[i][j] = h_inv[i][j] = h_old[i][j] = h_old_inv[i][j] = p_current[i][j] = p_target[i][j] =
        p_freq[i][j] = omega_dot[i][j] = omega_mass[i][j] = p_flag[i][j] = 0;
      p_period[i][j] = 1000;
      // TODO: if non-periodic...?
      need_scale[i][j] = true;
    }
  }

  if (strcmp(params[2], "x") == 0) {
    direction = 0;
  } else if (strcmp(params[2], "y") == 0) {
    direction = 1;
  } else if (strcmp(params[2], "z") == 0) {
    direction = 2;
  }
  if (!is_valid_real(params[3], &vs))
    PRINT_INPUT_ERROR("Wrong inputs for vs keyword.");
  if (!is_valid_real(params[4], &p_period[direction][direction]))
    PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
  p_flag[direction][direction] = 1;
  printf(
    "Perform msst by shock velocity %f km/s, barostat period %d timesteps.",
    vs,
    p_period[direction][direction]);
}

Ensemble_MSST_MTTK::~Ensemble_MSST_MTTK(void) { delete[] Q_p, eta_p_dot, eta_p_dotdot; }

void Ensemble_MSST_MTTK::init()
{
  for (int i = 0; i < atom->number_of_atoms; i++)
    total_mass += atom->cpu_mass[i];
  dt = time_step;
  dt2 = dt / 2;
  dt4 = dt / 4;
  dt8 = dt / 8;
  dt16 = dt / 16;
  Q_p = new double[pchain];
  eta_p_dot = new double[pchain + 1];
  eta_p_dotdot = new double[pchain];

  for (int n = 0; n < pchain; n++)
    Q_p[n] = eta_p_dot[n] = eta_p_dotdot[n] = 0;

  eta_p_dot[pchain] = 0;

  t_for_barostat = find_current_temperature();
  find_current_pressure();
  p0 = p_current[direction][direction];
  V0 = box->get_volume();
  // convert km/s to A/fs: 10e3 * 10e10  / 10e15 = 0.01
  // 1 km/s = 10 A/ps = 0.01A/fs
  vs *= 0.01;
  // convert A/fs to GPUMD unit
  vs *= TIME_UNIT_CONVERSION;
  printf("V0 is %f, p0 is %f.\n", V0, p0 * PRESSURE_UNIT_CONVERSION);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j]) {
        p_freq[i][j] = 1 / (p_period[i][j] * dt);
        if (p_freq_max < p_freq[i][j])
          p_freq_max = p_freq[i][j];
        omega_mass[i][j] =
          (atom->number_of_atoms + 1) * kB * t_for_barostat / (p_freq[i][j] * p_freq[i][j]);
      }
    }
  }
}

void Ensemble_MSST_MTTK::get_target_pressure()
{
  double V = box->get_volume();
  double p_msst = total_mass * (V0 - V) * vs * vs / (V0 * V0);
  p_target[direction][direction] = p0 + p_msst;
  // printf("%f %f %f\n", V0, V, p_msst * PRESSURE_UNIT_CONVERSION);
}

void Ensemble_MSST_MTTK::get_h_matrix_from_box()
{
  box->get_inverse();
  if (box->triclinic) {
    for (int x = 0; x < 3; x++) {
      for (int y = 0; y < 3; y++) {
        h[x][y] = box->cpu_h[y + x * 3];
        h_inv[x][y] = box->cpu_h[9 + y + x * 3];
      }
    }
  } else {
    for (int i = 0; i < 3; i++) {
      h[i][i] = box->cpu_h[i];
      h_inv[i][i] = box->cpu_h[9 + i];
    }
  }
}

void Ensemble_MSST_MTTK::copy_h_matrix_to_box()
{
  if (box->triclinic) {
    for (int x = 0; x < 3; x++) {
      for (int y = 0; y < 3; y++)
        box->cpu_h[y + x * 3] = h[x][y];
    }
  } else {
    for (int i = 0; i < 3; i++)
      box->cpu_h[i] = h[i][i];
  }
  box->get_inverse();
}

void Ensemble_MSST_MTTK::find_current_pressure()
{
  find_thermo();
  double t[8];
  thermo->copy_to_host(t, 8);
  p_current[0][0] = t[2];
  p_current[1][1] = t[3];
  p_current[2][2] = t[4];
  p_current[0][1] = p_current[1][0] = t[5];
  p_current[0][2] = p_current[2][0] = t[6];
  p_current[1][2] = p_current[2][1] = t[7];
}

void Ensemble_MSST_MTTK::nh_omega_dot()
{
  // Eq. (1) of Shinoda2004
  find_current_pressure();
  double f_omega, V;
  V = box->get_volume();
  f_omega = V * (p_current[direction][direction] - p_target[direction][direction]);
  f_omega /= omega_mass[direction][direction];
  if (V > V0 && f_omega > 0.0)
    f_omega = -f_omega;
  omega_dot[direction][direction] += f_omega * dt2;
}

void Ensemble_MSST_MTTK::propagate_box()
{
  // Eq. (1) of Shinoda2004
  // save old box
  box->get_inverse();
  get_h_matrix_from_box();
  std::copy(&h[0][0], &h[0][0] + 9, &h_old[0][0]);
  std::copy(&h_inv[0][0], &h_inv[0][0] + 9, &h_old_inv[0][0]);
  // change box, according to h_dot = omega_dot * h
  propagate_box_diagonal();
  scale_positions();
  copy_h_matrix_to_box();
}

void Ensemble_MSST_MTTK::propagate_box_diagonal()
{
  // TODO: fix point ?
  double expfac = exp(dt2 * omega_dot[direction][direction]);
  h[direction][direction] *= expfac;
}

void Ensemble_MSST_MTTK::find_thermo()
{
  Ensemble::find_thermo(
    false,
    box->get_volume(),
    *group,
    atom->mass,
    atom->potential_per_atom,
    atom->velocity_per_atom,
    atom->virial_per_atom,
    *thermo);
}

double Ensemble_MSST_MTTK::find_current_temperature()
{
  find_thermo();
  double t = 0;
  thermo->copy_to_host(&t, 1);
  return t;
}

void Ensemble_MSST_MTTK::nhc_press_integrate()
{

  int cell_dof; // DOF of cell
  double expfac, factor_eta_p;
  double kT;
  double ke_omega_current, ke_omega_target;

  kT = kB * t_for_barostat;

  double nkt = (atom->number_of_atoms + 1) * kT;

  omega_mass[direction][direction] =
    nkt / (p_freq[direction][direction] * p_freq[direction][direction]);

  Q_p[0] = kT / (p_freq_max * p_freq_max);
  for (int n = 1; n < pchain; n++)
    Q_p[n] = kT / (p_freq_max * p_freq_max);
  for (int n = 1; n < pchain; n++)
    eta_p_dotdot[n] = (Q_p[n - 1] * eta_p_dot[n - 1] * eta_p_dot[n - 1] - kT) / Q_p[n];

  cell_dof = 1;
  ke_omega_current = omega_mass[direction][direction] * omega_dot[direction][direction] *
                     omega_dot[direction][direction];

  ke_omega_target = cell_dof * kT;
  eta_p_dotdot[0] = (ke_omega_current - ke_omega_target) / Q_p[0];

  for (int n = pchain - 1; n >= 0; n--) {
    expfac = exp(-dt8 * eta_p_dot[n + 1]);
    eta_p_dot[n] = (eta_p_dot[n] * expfac + eta_p_dotdot[n] * dt4) * expfac;
  }

  factor_eta_p = exp(-dt2 * eta_p_dot[0]);
  omega_dot[direction][direction] *= factor_eta_p;

  ke_omega_current = omega_mass[direction][direction] * omega_dot[direction][direction] *
                     omega_dot[direction][direction];

  eta_p_dotdot[0] = (ke_omega_current - ke_omega_target) / Q_p[0];
  eta_p_dot[0] = (eta_p_dot[0] * expfac + eta_p_dotdot[0] * dt4) * expfac;

  for (int n = 1; n < pchain; n++) {
    expfac = exp(-dt8 * eta_p_dot[n + 1]);
    eta_p_dotdot[n] = (Q_p[n - 1] * eta_p_dot[n - 1] * eta_p_dot[n - 1] - kT) / Q_p[n];
    eta_p_dot[n] = (eta_p_dot[n] * expfac + eta_p_dotdot[n] * dt4) * expfac;
  }
}

static __global__ void gpu_scale_positions(
  int number_of_atoms,
  double hax,
  double hbx,
  double hcx,
  double hay,
  double hby,
  double hcy,
  double haz,
  double hbz,
  double hcz,
  double h_old_inv_ax,
  double h_old_inv_bx,
  double h_old_inv_cx,
  double h_old_inv_ay,
  double h_old_inv_by,
  double h_old_inv_cy,
  double h_old_inv_az,
  double h_old_inv_bz,
  double h_old_inv_cz,
  double* x,
  double* y,
  double* z)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    double old_pos[3] = {x[i], y[i], z[i]};
    double frac[3], new_pos[3];
    double h_old_inv[3][3] = {
      {h_old_inv_ax, h_old_inv_bx, h_old_inv_cx},
      {h_old_inv_ay, h_old_inv_by, h_old_inv_cy},
      {h_old_inv_az, h_old_inv_bz, h_old_inv_cz}};
    double h_new[3][3] = {{hax, hbx, hcx}, {hay, hby, hcy}, {haz, hbz, hcz}};
    // fractional position
    matrix_vector_multiply(h_old_inv, old_pos, frac);
    // new position
    matrix_vector_multiply(h_new, frac, new_pos);
    x[i] = new_pos[0];
    y[i] = new_pos[1];
    z[i] = new_pos[2];
  }
}

void Ensemble_MSST_MTTK::scale_positions()
{
  int n = atom->number_of_atoms;
  gpu_scale_positions<<<(n - 1) / 128 + 1, 128>>>(
    atom->number_of_atoms,
    h[0][0],
    h[0][1],
    h[0][2],
    h[1][0],
    h[1][1],
    h[1][2],
    h[2][0],
    h[2][1],
    h[2][2],
    h_old_inv[0][0],
    h_old_inv[0][1],
    h_old_inv[0][2],
    h_old_inv[1][0],
    h_old_inv[1][1],
    h_old_inv[1][2],
    h_old_inv[2][0],
    h_old_inv[2][1],
    h_old_inv[2][2],
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + n,
    atom->position_per_atom.data() + 2 * n);
}

static __global__ void gpu_nh_v_press(
  int number_of_particles,
  double time_step,
  double* vx,
  double* vy,
  double* vz,
  double omega_dot_ax,
  double omega_dot_bx,
  double omega_dot_cx,
  double omega_dot_ay,
  double omega_dot_by,
  double omega_dot_cy,
  double omega_dot_az,
  double omega_dot_bz,
  double omega_dot_cz)
{
  double dt4 = time_step / 4;
  double dt2 = time_step / 2;
  double factor_x = exp(-dt4 * omega_dot_ax);
  double factor_y = exp(-dt4 * omega_dot_by);
  double factor_z = exp(-dt4 * omega_dot_cz);

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    vx[i] *= factor_x;
    vy[i] *= factor_y;
    vz[i] *= factor_z;

    vx[i] += -dt2 * (vy[i] * omega_dot_bx + vz[i] * omega_dot_cx);
    vy[i] += -dt2 * (vx[i] * omega_dot_ay + vz[i] * omega_dot_cy);
    vz[i] += -dt2 * (vx[i] * omega_dot_az + vy[i] * omega_dot_bz);

    vx[i] *= factor_x;
    vy[i] *= factor_y;
    vz[i] *= factor_z;
  }
}

void Ensemble_MSST_MTTK::nh_v_press()
{
  int n = atom->number_of_atoms;
  gpu_nh_v_press<<<(n - 1) / 128 + 1, 128>>>(
    n,
    time_step,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + n,
    atom->velocity_per_atom.data() + 2 * n,
    omega_dot[0][0],
    omega_dot[0][1],
    omega_dot[0][2],
    omega_dot[1][0],
    omega_dot[1][1],
    omega_dot[1][2],
    omega_dot[2][0],
    omega_dot[2][1],
    omega_dot[2][2]);
}

void Ensemble_MSST_MTTK::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0) {
    init();
  }

  nhc_press_integrate();

  get_h_matrix_from_box();
  get_target_pressure();
  nh_omega_dot();
  nh_v_press();

  velocity_verlet_v();

  propagate_box();

  velocity_verlet_x();

  propagate_box();
}

void Ensemble_MSST_MTTK::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet_v();

  get_h_matrix_from_box();
  nh_v_press();

  nh_omega_dot();

  nhc_press_integrate();

  find_thermo();
}