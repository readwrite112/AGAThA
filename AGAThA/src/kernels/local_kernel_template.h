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



__global__ void gasal_local_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint4 *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_inter_row, int stretch, int zdrop, int W)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID

	int32_t i, j, k, m, l, y, u;
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
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	//--------------------------------------------


	//////////////////////////////////////////////
	int tx = threadIdx.x;
	const int const_warp_len = 8;
	int warp_len = const_warp_len;
	int warp_id = tx % warp_len; // id of warp in 
	int real_warp_id = tx % 32;
	int warp_num = tid / warp_len;
	int warp_per_kernel = (gridDim.x * blockDim.x) / warp_len; // number of warps. assume number of threads % warp_len == 0
	int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel ;
	
	// shared memory for intermediate values
	extern __shared__ int32_t shared_maxHH[];
	int job_per_query = max_query_len % warp_len ? (max_query_len / warp_len + 1) : max_query_len / warp_len;

	// start and end idx of sequences for each warp
	int job_start_idx = warp_num*job_per_warp;
	int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks;

	int32_t packed_target_batch_idx, packed_query_batch_idx, read_len, ref_len, query_batch_regs, target_batch_regs;

	const int packed_len = 8;


	int band_start, band_end, completed_band, stretch_start, stretch_end;
	int band_target, band_query;
	int total_diags;
	register uint32_t gpac, rpac; 
	short2* global_inter_col = (short2*)(global_inter_row+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_inter_col_p= (int32_t*)(global_inter_col+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_idx = (short2*)(global_inter_row+max_query_len*(blockDim.x/8)*gridDim.x*3);

	const int total_shm = packed_len*(stretch+1); 
	bool active, zdropped;

	int32_t* diag_maxHH = (int32_t*)(shared_maxHH+(threadIdx.x/32)*total_shm*32);
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(tx/32)*28;
	int diag_idx, temp, last_diag;
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);

	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	for (u = job_start_idx; u < job_end_idx; u++) {

		i = ((u&3)==0)? global_idx[n_tasks-(u>>2)-1].y: global_idx[u-(u>>2)-1].y;
				
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
			}
		}
		for (j = 0; j < job_per_query; j++) {
			l = j*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_inter_col[warp_num*max_query_len + l] =  l <= W? make_short2(k, k-_cudaGapOE):initHD;	
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

		if (warp_id==0) shared_job[(warp_num&3)] = total_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_target_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+read_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = i;

		same_threads = __match_any_sync(__activemask(), warp_num);

		
		__syncwarp();

		while (j < total_diags) {

			band_start = max(0, (j-query_batch_regs+1));
			band_start = max(band_start, (j*packed_len + packed_len-1+1 - W)/2/packed_len);
			band_end = min(target_batch_regs-1, j+stretch-1);
			band_end = min(band_end, ((j+stretch-1)*packed_len + packed_len-1 + W)/2/packed_len);
			completed_band = band_start;
			

			if (band_start > band_end) {
				zdropped = true;
			}

			while (!zdropped && completed_band <= band_end) {
				// while the entire chunk diag is not finished
				band_target = completed_band + warp_id;
				band_query = j - band_target;
				active = (band_target <= band_end);				
				
				if (active) {
					gidx = band_target << 3;
					ridx = band_query << 3;

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
						
						for (k = 28; k >= 0 && ridx < read_len; k -= 4) {
							uint32_t rbase = (rpac >> k) & 15;	//get a base from query_batch sequence
							//-----load intermediate values--------------
							HD = global_inter_row[warp_num*max_query_len + ridx];
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
							global_inter_row[warp_num*max_query_len + ridx] = HD;
							//---------------------------------------------

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

			__syncwarp();

			last_diag = (j+stretch)<<3;
			prev_maxHH = read_len+ref_len-1;

			if (!zdropped) {
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
			}
			
			__syncwarp();

			if (zdropped) {
				total_diags = j;
				if (warp_id==0) shared_job[(warp_num&3)] = total_diags;
			}

			if (warp_id==1) shared_job[20+(warp_num&3)] = maxHH;
			else if (warp_id==2) shared_job[24+(warp_num&3)] = (maxXY_y<<16) + maxXY_x;
 
			__syncwarp();

			

			j+=stretch;

			if (j >= total_diags) {

				if (!zdropped) {
					diag_idx = (j*packed_len)&(total_shm-1);
					for (k = j*packed_len, m = diag_idx; m < diag_idx+packed_len; m++, k++) {
						temp = __reduce_max_sync(same_threads, diag_maxHH[(m<<5)+real_warp_id]);
						if ((temp>>16) > maxHH) {				
							maxHH = temp>>16;
							maxXY_y = (temp&65535);
							maxXY_x = k-maxXY_y; 
						} else if ( (temp&65535) >= maxXY_y && (k-(temp&65535)) >= maxXY_x) {
							int tl =  (temp&65535) - maxXY_y, ql = (k-(temp&65535)) - maxXY_x, l;
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
					device_res->aln_score[i] = maxHH;//copy the max score to the output array in the GPU mem
					device_res->query_batch_end[i] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
					device_res->target_batch_end[i] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem
				}

				for (m=0; m<4;m++) {
					if (shared_job[m]> j) {
						total_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						packed_target_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						read_len = ref_len&65535;
						ref_len = ref_len>>16;
						i = shared_job[16+m];
						query_batch_regs = (read_len >> 3) + (read_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
						target_batch_regs = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
						
						
						maxHH = shared_job[20+m];
						maxXY_y = shared_job[24+m];
						maxXY_x = maxXY_y&65535;
						maxXY_y = maxXY_y>>16;
						
						zdropped = false;

						for (m = 0; m < total_shm; m++) {
							diag_maxHH[(m<<5)+real_warp_id]=INT_MIN;
						}
						
						break;
					}
				}

			}

			__syncwarp();
			
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (tx % 32))&same_threads))-1;
			
			__syncwarp();

		}

		__syncwarp();

		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		i = shared_job[16+(warp_num&3)];

		__syncwarp();



	}
	
	return;


}


__global__ void saloba_sort(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int n_tasks, uint32_t max_query_len, short2 *global_inter_row)
{

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID

	uint32_t read_len, ref_len, query_batch_regs, target_batch_regs;

	short2* global_idx = (short2*)(global_inter_row+max_query_len*(blockDim.x/8)*gridDim.x*3);

	if (tid < n_tasks) {

		read_len = query_batch_lens[tid];
		ref_len = target_batch_lens[tid];
		query_batch_regs = (read_len >> 3) + (read_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		target_batch_regs = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);

		global_idx[tid] = make_short2((target_batch_regs + query_batch_regs-1), tid);


	}
	
	return;


}
#endif
