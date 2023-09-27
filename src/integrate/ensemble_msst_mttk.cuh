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

#pragma once
#include "ensemble.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include <math.h>

class Ensemble_MSST_MTTK : public Ensemble
{
public:
  Ensemble_MSST_MTTK(const char** params, int num_params);
  virtual ~Ensemble_MSST_MTTK(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo);

protected:
  void init();
  void nhc_press_integrate();
  void get_target_pressure();
  double find_current_temperature();
  void find_current_pressure();
  void find_thermo();
  void get_h_matrix_from_box();
  void copy_h_matrix_to_box();
  void nh_omega_dot();
  void propagate_box();
  void propagate_box_diagonal();
  void scale_positions();
  void nh_v_press();

  int direction = -1;
  double total_mass = 0;
  double V0 = 0, p0 = 0;
  double vs = 0;
  // When nph, there is no target temperature. So we use the temperature of kinetic energy.
  double t_for_barostat = 0;
  // the 3x3 matric of cell parameters
  double h[3][3], h_inv[3][3], h_old[3][3], h_old_inv[3][3];

  double p_current[3][3], p_target[3][3];
  double p_period[3][3], p_freq[3][3];
  double p_freq_max = 0;
  double omega_dot[3][3], omega_mass[3][3];

  bool p_flag[3][3]; // 1 if control P on this dim, 0 if not
  bool need_scale[3][3];
  double dt, dt2, dt4, dt8, dt16;

  double *Q_p, *eta_p_dot, *eta_p_dotdot;
  double factor_eta = 0;
  const double kB = 8.617333262e-5;
  // length of Nose-Hoover chain
  int pchain = 4;
};