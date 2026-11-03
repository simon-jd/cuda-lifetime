/* Stand-in for the OpenMP runtime header, which this LLVM build does not
 * install: the runtime routines RAFT's and cuVS's headers call, declared as
 * OpenMP 5.2 declares them. Without -fopenmp the analyzer ignores the
 * pragmas and needs only these declarations. */
#ifndef CUDA_LIFETIME_OMP_STAND_IN_H
#define CUDA_LIFETIME_OMP_STAND_IN_H
#ifdef __cplusplus
extern "C" {
#endif
int omp_get_max_threads(void);
int omp_get_num_procs(void);
int omp_get_num_threads(void);
int omp_get_thread_num(void);
#ifdef __cplusplus
}
#endif
#endif
