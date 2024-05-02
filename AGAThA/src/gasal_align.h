#ifndef __GASAL_ALIGN_H__
#define __GASAL_ALIGN_H__

void gasal_copy_subst_scores(gasal_subst_scores *subst);

void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, Parameters *params, uint32_t maximum_sequence_length, short2* global_inter_row, int stretch, int zdrop, int W);

inline void gasal_kernel_launcher(int32_t N_BLOCKS, int32_t BLOCKDIM, algo_type algo, comp_start start, gasal_gpu_storage_t *gpu_storage, int32_t actual_n_alns, int32_t k_band, uint32_t maximum_sequence_length, short2* global_inter_row, int stretch, int zdrop, int W);

int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage);

#endif
