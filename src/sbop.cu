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
Smooth bond-order potential
    Written by Zheyong Fan.
    This is a new potential I proposed. 
------------------------------------------------------------------------------*/


#include "sbop.cuh"
#include "mic.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE_FORCE 64
#define ONE_OVER_16      0.0625
#define NINE_OVER_16     0.5625

// Easy labels for indexing
#define A            0
#define Q            1
#define LAMBDA       2
#define B            3
#define MU           4
#define B2           5
#define MU2          6
#define BETA         7
#define H            8
#define ALPHA        9
#define R1           10
#define R2           11
#define PI_FACTOR1   12
#define PI_FACTOR3   13
#define NUM_PARAMS   14


SBOP::SBOP(FILE *fid, Atom* atom, int num_of_types)
{
    num_types = num_of_types;
    printf("Use SBOP (%d-element) potential.\n", num_types);
    int n_entries = num_types*num_types*num_types;
    double *cpu_para;
    MY_MALLOC(cpu_para, double, n_entries*NUM_PARAMS);

    const char err[] = "Error: Illegal SBOP parameter.";
    rc = 0.0;
    int count;
    double a, q, lambda, b, mu, b2, mu2, beta, h, alpha, r1, r2;
    for (int i = 0; i < n_entries; i++)
    {
        count = fscanf
        (
            fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
            &a, &q, &lambda, &b, &mu, &b2, &mu2, &beta, &h, &alpha, &r1, &r2
        );
        if (count != 12) 
            {printf("Error: reading error for SBOP potential.\n"); exit(1);}

        if (a < 0.0)
            {printf("%s A must be >= 0.\n",err); exit(1);}
        if (q < 0.0)
            {printf("%s Q must be >= 0.\n",err); exit(1);}
        if (lambda < 0.0)
            {printf("%s lambda must be >= 0.\n",err); exit(1);}
        if (b < 0.0)
            {printf("%s B must be >= 0.\n",err); exit(1);}
        if(mu < 0.0)
            {printf("%s mu must be >= 0.\n",err); exit(1);}
        if(b2 < 0.0)
            {printf("%s B2 must be >= 0.\n",err); exit(1);}
        if(mu2 < 0.0)
            {printf("%s mu2 must be >= 0.\n",err); exit(1);}
        if(beta < 0.0)
            {printf("%s beta must be >= 0.\n",err); exit(1);}
        if(h < -1.0 || h > 1.0)
            {printf("%s |h| must be <= 1.\n",err); exit(1);}
        if (alpha < 0.0)
            {printf("%s alpha must be >= 0.\n",err); exit(1);}
        if(r1 < 0.0)
            {printf("%s R1 must be >= 0.\n",err); exit(1);}
        if(r2 <= 0.0)
            {printf("%s R2 must be > 0.\n",err); exit(1);}
        if(r2 <= r1)
            {printf("%s R2-R1 must be > 0.\n",err); exit(1);}

        cpu_para[i*NUM_PARAMS + A] = a;
        cpu_para[i*NUM_PARAMS + Q] = q;
        cpu_para[i*NUM_PARAMS + LAMBDA] = lambda;
        cpu_para[i*NUM_PARAMS + B] = b;
        cpu_para[i*NUM_PARAMS + MU] = mu;
        cpu_para[i*NUM_PARAMS + B2] = b2;
        cpu_para[i*NUM_PARAMS + MU2] = mu2;
        cpu_para[i*NUM_PARAMS + BETA] = beta;
        cpu_para[i*NUM_PARAMS + H] = h;
        cpu_para[i*NUM_PARAMS + ALPHA] = alpha;
        cpu_para[i*NUM_PARAMS + R1] = r1;
        cpu_para[i*NUM_PARAMS + R2] = r2;
        cpu_para[i*NUM_PARAMS + PI_FACTOR1] = PI / (r2 - r1);
        cpu_para[i*NUM_PARAMS + PI_FACTOR3] = 3.0 * PI / (r2 - r1);
        rc = r2 > rc ? r2 : rc;
    }

    int num_of_neighbors = (atom->neighbor.MN < 20) ? atom->neighbor.MN : 20;
    int memory1 = sizeof(double)* atom->N * num_of_neighbors;
    int memory2 = sizeof(double)* n_entries * NUM_PARAMS;
    CHECK(cudaMalloc((void**)&sbop_data.b,    memory1));
    CHECK(cudaMalloc((void**)&sbop_data.bp,   memory1));
    CHECK(cudaMalloc((void**)&sbop_data.f12x, memory1));
    CHECK(cudaMalloc((void**)&sbop_data.f12y, memory1));
    CHECK(cudaMalloc((void**)&sbop_data.f12z, memory1));
    CHECK(cudaMalloc((void**)&para, memory2));
    CHECK(cudaMemcpy(para, cpu_para, memory2, cudaMemcpyHostToDevice));
    MY_FREE(cpu_para);
}


SBOP::~SBOP(void)
{
    CHECK(cudaFree(sbop_data.b));
    CHECK(cudaFree(sbop_data.bp));
    CHECK(cudaFree(sbop_data.f12x));
    CHECK(cudaFree(sbop_data.f12y));
    CHECK(cudaFree(sbop_data.f12z));
    CHECK(cudaFree(para));
}


static __device__ void find_fr_and_frp
(int i, const double* __restrict__ para, double d12, double &fr, double &frp)
{
    double exp_factor = LDG(para, i + A) * exp(- LDG(para, i + LAMBDA) * d12);
    double d_inv = 1.0 / d12;
    fr = (1.0 + LDG(para, i + Q) * d_inv) * exp_factor;
    frp = - LDG(para, i + LAMBDA) * fr;
    frp -= LDG(para, i + Q) * d_inv * d_inv * exp_factor;
}


static __device__ void find_fa_and_fap
(int i, const double* __restrict__ para, double d12, double &fa, double &fap)
{
    fa  = LDG(para, i + B) * exp(- LDG(para, i + MU) * d12);
    fap = - LDG(para, i + MU) * fa;
    double tmp =  LDG(para, i + B2) * exp(- LDG(para, i + MU2) * d12);
    fa += tmp;
    fap -= LDG(para, i + MU2) * tmp;
}


static __device__ void find_fa
(int i, const double* __restrict__ para, double d12, double &fa)
{
    fa = LDG(para, i + B) * exp(- LDG(para, i + MU) * d12);
    fa += LDG(para, i + B2) * exp(- LDG(para, i + MU2) * d12);
}


static __device__ void find_fc_and_fcp
(int i, const double* __restrict__ para, double d12, double &fc, double &fcp)
{
    if (d12 < LDG(para, i + R1)){fc = 1.0; fcp = 0.0;}
    else if (d12 < LDG(para, i + R2))
    {
        double tmp = d12 - LDG(para, i + R1);
        double pi_factor1 = LDG(para, i + PI_FACTOR1);
        double pi_factor3 = LDG(para, i + PI_FACTOR3);

        fc = NINE_OVER_16 * cos(pi_factor1 * tmp)
           - ONE_OVER_16  * cos(pi_factor3 * tmp)
           + 0.5;

        fcp = sin(pi_factor3 * tmp) * pi_factor3 * ONE_OVER_16
            - sin(pi_factor1 * tmp) * pi_factor1 * NINE_OVER_16;
    }
    else {fc  = 0.0; fcp = 0.0;}
}


static __device__ void find_fc
(int i, const double* __restrict__ para, double d12, double &fc)
{
    if (d12 < LDG(para, i + R1)) {fc  = 1.0;}
    else if (d12 < LDG(para, i + R2))
    {
        double tmp = d12 - LDG(para, i + R1);
        fc = NINE_OVER_16 * cos(LDG(para, i + PI_FACTOR1) * tmp)
           - ONE_OVER_16  * cos(LDG(para, i + PI_FACTOR3) * tmp)
           + 0.5;
    }
    else {fc  = 0.0;}
}


static __device__ void find_g_and_gp
(int i, const double* __restrict__ para, double cos, double &g, double &gp)
{
    double tmp = cos - LDG(para, i + H);
    g  = tmp * tmp;
    gp = 2.0 * tmp;
}


static __device__ void find_g
(int i, const double* __restrict__ para, double cos, double &g)
{
    double tmp = cos - LDG(para, i + H);
    g = tmp * tmp;
}


static __device__ void find_e_and_ep
(
    int i, const double* __restrict__ para, 
    double d12, double d13, double &e, double &ep
)
{
    if (LDG(para, i + ALPHA) < 1.0e-15){ e = 1.0; ep = 0.0;}
    else
    {
        double r = d12 - d13;
        e = exp(LDG(para, i + ALPHA) * r * r * r);
        ep = LDG(para, i + ALPHA) * 3.0 * r * r * e;
    }
}


static __device__ void find_e
(int i, const double* __restrict__ para, double d12, double d13, double &e)
{
    if (LDG(para, i + ALPHA) < 1.0e-15){ e = 1.0;}
    else
    {
        double r = d12 - d13;
        e = exp(LDG(para, i + ALPHA) * r * r * r);
    }
}


// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_step1
(
    int number_of_particles, int N1, int N2, 
    int triclinic, int pbc_x, int pbc_y, int pbc_z,
    int num_types, int* g_neighbor_number, int* g_neighbor_list, int* g_type,
    const double* __restrict__ para,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const double* __restrict__ g_box,
    double* g_b, double* g_bp
)
{
    // start from the N1-th atom
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    // to the (N2-1)-th atom
    if (n1 >= N1 && n1 < N2)
    {
        int num_types2 = num_types * num_types;
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int type2 = g_type[n2];
            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double zeta = 0.0;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3];
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, 
                    x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12*d13);
                double fc_ijk_13, g_ijk, e_ijk_12_13;
                int ijk = type1 * num_types2 + type2 * num_types + type3;
                if (d13 > LDG(para, ijk*NUM_PARAMS + R2)) {continue;}
                find_fc(ijk*NUM_PARAMS, para, d13, fc_ijk_13);
                find_g(ijk*NUM_PARAMS, para, cos123, g_ijk);
                find_e(ijk*NUM_PARAMS, para, d12, d13, e_ijk_12_13);
                zeta += fc_ijk_13 * g_ijk * e_ijk_12_13;
            }
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            double beta = LDG(para, ijj*NUM_PARAMS + BETA);
            double b_ijj = 1.0 / sqrt(1.0 + beta * zeta);
            g_b[i1 * number_of_particles + n1]  = b_ijj;
            g_bp[i1 * number_of_particles + n1] = - 0.5 * beta * b_ijj 
                                                / (1.0 + beta * zeta);
        }
    }
}


// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void find_force_step2
(
    int number_of_particles, int N1, int N2, 
    int triclinic, int pbc_x, int pbc_y, int pbc_z,
    int num_types, int *g_neighbor_number, int *g_neighbor_list, int *g_type,
    const double* __restrict__ para,
    const double* __restrict__ g_b,
    const double* __restrict__ g_bp,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const double* __restrict__ g_box,
    double *g_potential, double *g_f12x, double *g_f12y, double *g_f12z
)
{
    // start from the N1-th atom
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    // to the (N2-1)-th atom
    if (n1 >= N1 && n1 < N2)
    {
        int num_types2 = num_types * num_types;
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        double pot_energy = 0.0;
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2];

            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = ONE / d12;
            double fc_ijj_12, fcp_ijj_12;
            double fa_ijj_12, fap_ijj_12, fr_ijj_12, frp_ijj_12;
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            find_fc_and_fcp(ijj*NUM_PARAMS, para, d12, fc_ijj_12, fcp_ijj_12);
            find_fa_and_fap(ijj*NUM_PARAMS, para, d12, fa_ijj_12, fap_ijj_12);
            find_fr_and_frp(ijj*NUM_PARAMS, para, d12, fr_ijj_12, frp_ijj_12);

            // (i,j) part
            double b12 = LDG(g_b, index);
            double factor3=(fcp_ijj_12*(fr_ijj_12-b12*fa_ijj_12)+
                          fc_ijj_12*(frp_ijj_12-b12*fap_ijj_12))*d12inv;
            double f12x = x12 * factor3 * 0.5;
            double f12y = y12 * factor3 * 0.5;
            double f12z = z12 * factor3 * 0.5;

            // accumulate potential energy
            pot_energy += fc_ijj_12 * (fr_ijj_12 - b12 * fa_ijj_12) * 0.5;

            // (i,j,k) part
            double bp12 = LDG(g_bp, index);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;
                int n3 = g_neighbor_list[index_2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, 
                    x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double fc_ikk_13, fc_ijk_13, fa_ikk_13, fc_ikj_12, fcp_ikj_12;
                int ikj = type1 * num_types2 + type3 * num_types + type2;
                int ikk = type1 * num_types2 + type3 * num_types + type3;
                int ijk = type1 * num_types2 + type2 * num_types + type3;
                find_fc(ikk*NUM_PARAMS, para, d13, fc_ikk_13);
                find_fc(ijk*NUM_PARAMS, para, d13, fc_ijk_13);
                find_fa(ikk*NUM_PARAMS, para, d13, fa_ikk_13);
                find_fc_and_fcp(ikj*NUM_PARAMS, para, d12,
                                	fc_ikj_12, fcp_ikj_12);
                double bp13 = LDG(g_bp, index_2);
                double one_over_d12d13 = ONE / (d12 * d13);
                double cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                double cos123_over_d12d12 = cos123*d12inv*d12inv;
                double g_ijk, gp_ijk;
                find_g_and_gp(ijk*NUM_PARAMS, para, cos123, g_ijk, gp_ijk);

                double g_ikj, gp_ikj;
                find_g_and_gp(ikj*NUM_PARAMS, para, cos123, g_ikj, gp_ikj);

                // exp with d12 - d13
                double e_ijk_12_13, ep_ijk_12_13;
                find_e_and_ep(ijk*NUM_PARAMS, para, d12, d13,
                                	e_ijk_12_13, ep_ijk_12_13);

                // exp with d13 - d12
                double e_ikj_13_12, ep_ikj_13_12;
                find_e_and_ep(ikj*NUM_PARAMS, para, d13, d12,
                                	e_ikj_13_12, ep_ikj_13_12);

                // derivatives with cosine
                double dc=-fc_ijj_12*bp12*fa_ijj_12*fc_ijk_13*gp_ijk*e_ijk_12_13+
                        -fc_ikj_12*bp13*fa_ikk_13*fc_ikk_13*gp_ikj*e_ikj_13_12;
                // derivatives with rij
                double dr=(-fc_ijj_12*bp12*fa_ijj_12*fc_ijk_13*g_ijk*ep_ijk_12_13 +
                  (-fcp_ikj_12*bp13*fa_ikk_13*g_ikj*e_ikj_13_12 +
                  fc_ikj_12*bp13*fa_ikk_13*g_ikj*ep_ikj_13_12)*fc_ikk_13)*d12inv;
                double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * dr + dc * cos_d)*0.5;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * dr + dc * cos_d)*0.5;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * dr + dc * cos_d)*0.5;
            }
            g_f12x[index] = f12x; g_f12y[index] = f12y; g_f12z[index] = f12z;
        }
        // save potential
        g_potential[n1] += pot_energy;
    }
}


// Wrapper of force evaluation for the SBOP potential
void SBOP::compute(Atom *atom, Measure *measure)
{
    int N = atom->N;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    int triclinic = atom->box.triclinic;
    int pbc_x = atom->box.pbc_x;
    int pbc_y = atom->box.pbc_y;
    int pbc_z = atom->box.pbc_z;
    int *NN = atom->NN_local;
    int *NL = atom->NL_local;
    int *type = atom->type_local;
    double *x = atom->x;
    double *y = atom->y;
    double *z = atom->z;
    double *box = atom->box.h;
    double *pe = atom->potential_per_atom;

    // special data for SBOP potential
    double *f12x = sbop_data.f12x;
    double *f12y = sbop_data.f12y;
    double *f12z = sbop_data.f12z;
    double *b    = sbop_data.b;
    double *bp   = sbop_data.bp;

    // pre-compute the bond order functions and their derivatives
    find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, triclinic, pbc_x, pbc_y, pbc_z, num_types,
        NN, NL, type, para, x, y, z, box, b, bp
    );
    CUDA_CHECK_KERNEL

    // pre-compute the partial forces
    find_force_step2<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, triclinic, pbc_x, pbc_y, pbc_z, num_types,
        NN, NL, type, para, b, bp, x, y, z, box, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL

    // the final step: calculate force and related quantities
    find_properties_many_body(atom, measure, NN, NL, f12x, f12y, f12z);
}
