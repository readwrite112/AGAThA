#ifndef __LOCAL_KERNEL_TEMPLATE__
#define __LOCAL_KERNEL_TEMPLATE__


// This old core provides the same result as the currently LOCAL core, but lacks some optimization. Left for historical / comparative purposes.
#define CORE_LOCAL_DEPRECATED_COMPUTE() \
		uint32_t gbase = (gpac >> l) & 15;/*get a base from target_batch sequence */ \
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);/* check equality of rbase and gbase */ \
		f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */ \
		h[m] = p[m] + subScore; /*score if rbase is aligned to gbase*/ \
		h[m] = max(h[m], f[m]); \
		h[m] = max(h[m], 0); \
		e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);/*whether to introduce or extend a gap in target_batch sequence */\
		h[m] = max(h[m], e); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1];

#define CORE_LOCAL_COMPUTE() \
		if (ridx + W < gidx + m-1 || ridx - W > gidx + m-1) { \
			p[m] = h[m-1]; \
		} else { \
			uint32_t gbase = (gpac >> l) & 15;\
			DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase) \
			int32_t tmp_hm = p[m] + subScore; \
			h[m] = max(tmp_hm, f[m]); \
			h[m] = max(h[m], e); \
			f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
			e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
			p[m] = h[m-1]; \
			diag_idx = ((gidx + m-1+ridx)&(total_shm-1))<<5;\
			diag_maxHH[real_warp_id+diag_idx] = max(diag_maxHH[real_warp_id+diag_idx], (h[m]<<16) +gidx+ m-1);\
		}
		
		
#define CORE_LOCAL_COMPUTE_ORIGINAL() \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		p[m] = h[m-1]; \
		diag_idx = ((gidx + m-1+ridx)&(total_shm-1))<<5;\
		diag_maxHH[real_warp_id+diag_idx] = max(diag_maxHH[real_warp_id+diag_idx], (h[m]<<16) +gidx+ m-1);\

#define CORE_LOCAL_COMPUTE_START() \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1]; \

#define CORE_LOCAL_COMPUTE_TB(direction_reg) \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		uint32_t m_or_x = tmp_hm >= p[m] ? 0 : 1;\
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		direction_reg |= h[m] == tmp_hm ? m_or_x << (28 - ((m - 1) << 2)) : (h[m] == f[m] ? (uint32_t)3 << (28 - ((m - 1) << 2)) : (uint32_t)2 << (28 - ((m - 1) << 2)));\
		direction_reg |= (tmp_hm - _cudaGapOE) > (f[m] - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (31 - ((m - 1) << 2));\
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		direction_reg |= (tmp_hm - _cudaGapOE) > (e - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (30 - ((m - 1) << 2));\
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1]; \





/* typename meaning : 
    - T is the algorithm type (LOCAL, MICROLOCAL)
    - S is WITH_ or WIHTOUT_START
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/
template <typename T, typename S, typename B>
__global__ void gasal_local_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint4 *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_inter_row, int stretch)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	//if (tid >= n_tasks) return;

	int32_t i, j, k, m, l, y;
	int32_t e;

    int32_t maxHH = 0; //initialize the maximum score to zero
	int32_t maxXY_y = 0; 

    int32_t prev_maxHH = 0;
    int32_t maxXY_x = 0;


	int32_t subScore;

	int32_t gidx, ridx;
	short2 HD;
	short2 initHD = make_short2(MINUS_INF2, MINUS_INF2);
	
	//-----arrays for saving intermediate values------
	//short2 global[MAX_QUERY_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	//--------------------------------------------


	//////////////////////////////////////////////
	int tx = threadIdx.x;
	int warp_len = 8;
	int warp_id = tx % warp_len; // id of warp in 
	int real_warp_id = tx % 32;
	int warp_num = tid / warp_len;
	int warp_per_kernel = (gridDim.x * blockDim.x) / warp_len; // number of warps. assume number of threads % warp_len == 0
	int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel ;
	//int warp_per_block = blockDim.x / warp_len; // number of warps in a block 
	
	// shared memory for intermediate values
	//extern __shared__ short2 inter_row[];	//TODO: could use global mem instead
	extern __shared__ int32_t shared_maxHH[];
	int job_per_query = max_query_len % warp_len ? (max_query_len / warp_len + 1) : max_query_len / warp_len;

	// start and end idx of sequences for each warp
	int job_start_idx = warp_num*job_per_warp;
	int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks;

	int32_t packed_target_batch_idx, packed_query_batch_idx, read_len, ref_len, query_batch_regs, target_batch_regs;

	const int packed_len = 8;
	//const int shared_len = 32;
	//int iter_num;
	int W = 751;
	int band_start, band_end, completed_band, stretch_start, stretch_end;
	int band_target, band_query;
	int total_diags;
	register uint32_t gpac, rpac; 
	short2* global_inter_col = (short2*)(global_inter_row+max_query_len*(blockDim.x/8)*gridDim.x);
	//int32_t* saved_p = (int32_t*)(global_inter_col+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_inter_col_p= (int32_t*)(global_inter_col+max_query_len*(blockDim.x/8)*gridDim.x);

	const int total_shm = packed_len*(stretch+1); 
	bool active, zdropped;

	int32_t* diag_maxHH = (int32_t*)(shared_maxHH+(threadIdx.x/32)*total_shm*32);
	int diag_idx, temp, last_diag;
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	int zdrop = 400;

	for (i = job_start_idx; i < job_end_idx; i++) {
		
		// get target and query seq
		packed_target_batch_idx = target_batch_offsets[i] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[i] >> 3;//starting index of the query_batch sequence
		read_len = query_batch_lens[i];
		ref_len = target_batch_lens[i];
		query_batch_regs = (read_len >> 3) + (read_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		target_batch_regs = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// fill with initial value
		for (j = 0; j < job_per_query; j++) {
			l = j*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_inter_row[warp_num*max_query_len + l] =  l <= W? make_short2(k, k-_cudaGapOE):initHD;	
				//inter_row[warp_block_id*512 + l] = global_inter_row[warp_num*max_query_len + l];	
			}
		}
		for (j = 0; j < job_per_query; j++) {
			l = j*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_inter_col[warp_num*max_query_len + l] =  l <= W? make_short2(k, k-_cudaGapOE):initHD;	
				//inter_row[warp_block_id*512 + l] = global_inter_row[warp_num*max_query_len + l];	
			}
		}
		
		for (j = 0; j < job_per_query; j++) {
			l = j*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_inter_col_p[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= W? k: MINUS_INF2; 	
			}
		}
		

		for (m = 0; m < total_shm; m++) {
			diag_maxHH[real_warp_id + m*32] = INT_MIN;
		}

		__syncwarp();

		maxHH = 0; //initialize the maximum score to zero
		maxXY_y = 0; 

    	prev_maxHH = 0;
    	maxXY_x = 0;
		zdropped = false;

		j = 0; //chunk
		total_diags = target_batch_regs + query_batch_regs-1; //chunk


		while (j < total_diags) {
			// start of a new chunk diag
			/*
			band_start = (j*packed_len + packed_len-1+1 - W);
			band_start = band_start > 0? (band_start>>1)/packed_len:0;
			band_start = min(band_start, query_batch_regs-1);
			band_end = min(((j*packed_len + packed_len-1 + W)>>1)/packed_len, target_batch_regs-1);
			band_end = min(band_end, j);
			*/
			band_start = max(0, (j-query_batch_regs+1));
			band_start = max(band_start, (j*packed_len + packed_len-1+1 - W)/2/packed_len);
			band_end = min(target_batch_regs-1, j+stretch-1);
			band_end = min(band_end, ((j+stretch-1)*packed_len + packed_len-1 + W)/2/packed_len);
			completed_band = band_start;
			

			if (band_start > band_end) break;

			while (completed_band <= band_end) {
				// while the entire chunk diag is not finished
				band_target = completed_band + warp_id;
				band_query = j - band_target;
				active = (band_target <= band_end);				
				
				if (active) {
					gidx = band_target << 3;
					ridx = band_query << 3;
					
					//DEBUG = min(DEBUG, band_start);
					// if the current thread is within bounds
					// read packed bps

					// read scores 
					// TODO: edit score reading in a more efficient way 

					p[1] = global_inter_col_p[warp_num*max_query_len + band_target];

					
					for (m = 1; m < 9; m++) {
						if ( (gidx + m-1) < ref_len) {
							HD = global_inter_col[warp_num*max_query_len + gidx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					stretch_start = (max(0, (band_target*packed_len - W)))/packed_len;
					stretch_end = min( query_batch_regs-1, ( (band_target*packed_len + packed_len -1 + W )) /packed_len );
					gpac = packed_target_batch[packed_target_batch_idx + band_target];
				}
					
					
				for (y = 0; y < stretch; y++) {
					if (active && stretch_start <= band_query && band_query <=stretch_end) {
						
						rpac = packed_query_batch[packed_query_batch_idx + band_query]; 
						ridx = band_query << 3;

						//p[1] = saved_p[warp_num*max_query_len + band_query];


						
						for (k = 28; k >= 0 && ridx < read_len; k -= 4) {
							uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
							//-----load intermediate values--------------
							//HD = inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx];
							//HD = global_inter_row[warp_num*max_query_len + read_iter*packed_len + ridx];
							HD = global_inter_row[warp_num*max_query_len + ridx];
							//HD = global_inter_row[warp_num*max_query_len + ridx];
							//p[1] = HD.x;
							h[0] = HD.x;
							e = HD.y;
							
							if (band_query == stretch_start || band_query == stretch_end) {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_LOCAL_COMPUTE();
								}
							} else {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_LOCAL_COMPUTE_ORIGINAL();
								}
							}
							
							//----------save intermediate values------------
							HD.x = h[m-1];
							HD.y = e;
							//inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx] = HD;
							//global_inter_row[warp_num*max_query_len + read_iter*packed_len + ridx] = HD;
							global_inter_row[warp_num*max_query_len + ridx] = HD;
							//---------------------------------------------

							//TODO: remove max value calculation
							//maxXY_x = (prev_maxHH < maxHH) ? ridx+read_iter*8 : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
							ridx++;
							//-------------------------------------------------------

						}

					}
					

					band_query++;
					
				}
				

					// write scores 
					// TODO: edit score reading in a more efficient way 
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( gidx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_inter_col[warp_num*max_query_len + gidx + m-1] = HD;
						}
						
					}

					global_inter_col_p[warp_num*max_query_len + band_target] = p[1];
					
					
				}
				

				completed_band+=warp_len;
			}
			/*
			for (int c = 0; c < query_batch_regs; c+=warp_len) {
				if (c+warp_id+1<query_batch_regs) {
					saved_p[warp_num*max_query_len + c + warp_id+1] = (global_inter_row[warp_num*max_query_len+(c+warp_id+1)*packed_len-1]).x;
				}
			}
			*/
			__syncwarp();

			last_diag = (j+stretch)<<3;
			prev_maxHH = read_len+ref_len-1;

			for (diag_idx = j<<3; diag_idx < last_diag; diag_idx++) {
				if (diag_idx <prev_maxHH) {
					m = diag_idx&(total_shm-1);
					temp = __reduce_max_sync(same_threads, diag_maxHH[(m<<5)+real_warp_id]);
					if ((temp>>16) > maxHH) {				
						maxHH = temp>>16;
						maxXY_y = (temp&65535);
						maxXY_x = diag_idx-maxXY_y; 
					} else if ( (temp&65535) >= maxXY_y && (diag_idx-(temp&65535)) >= maxXY_x) {
						int tl =  (temp&65535) - maxXY_y, ql = (diag_idx-(temp&65535)) - maxXY_x, l;
						l = tl > ql? tl - ql : ql - tl;
						if (zdrop >= 0 && maxHH - (temp>>16) > zdrop + l*_cudaGapExtend) {
							
							zdropped = true;
							break;
							
						}
					}
					diag_maxHH[(m<<5)+real_warp_id]=INT_MIN;
				}
			}
			

			if (zdropped) break;

			/*
			swap = global_inter_row_old;
			global_inter_row_old = global_inter_row;
			global_inter_row = swap;
			*/

			__syncwarp();
			

			j+=stretch;

		}

		__syncwarp();

		if (!zdropped) {
			last_diag = read_len+ref_len-1;

			for (diag_idx = j<<3; diag_idx < last_diag; diag_idx++) {
				m = diag_idx&(total_shm-1);
				temp = __reduce_max_sync(same_threads, diag_maxHH[(m<<5)+real_warp_id]);
				if ((temp>>16) > maxHH) {				
					maxHH = temp>>16;
					maxXY_y = (temp&65535);
					maxXY_x = diag_idx-maxXY_y; 
				} else if ( (temp&65535) >= maxXY_y && (diag_idx-(temp&65535)) >= maxXY_x) {
					int tl =  (temp&65535) - maxXY_y, ql = (diag_idx-(temp&65535)) - maxXY_x, l;
					l = tl > ql? tl - ql : ql - tl;
					if (zdrop >= 0 && maxHH - (temp>>16) > zdrop + l*_cudaGapExtend) {
						
						zdropped = true;
						break;
						
					}
				}
				diag_maxHH[(m<<5)+real_warp_id]=INT_MIN;
			}
		}
		
		if (warp_id==0) {
			device_res->aln_score[i] =  maxHH;//copy the max score to the output array in the GPU mem
			device_res->query_batch_end[i] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
			device_res->target_batch_end[i] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem
		}

		__syncwarp();


	}
	



	///////////////////////////////////////////////

	



	return;


}
#endif
