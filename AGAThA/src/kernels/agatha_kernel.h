#ifndef __AGATHA_KERNEL__
#define __AGATHA_KERNEL__


// This old core provides the same result as the currently LOCAL core, but lacks some optimization. Left for historical / comparative purposes.
// Deprecated code from GASAL2 (left as reference)
#define CORE_LOCAL_DEPRECATED_COMPUTE() \
		uint32_t rbase = (packed_ref_literal >> l) & 15;/*get a base from target_batch sequence */ \
		DEV_GET_SUB_SCORE_LOCAL(temp_score, qbase, rbase);/* check equality of qbase and rbase */ \
		f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */ \
		h[m] = p[m] + temp_score; /*score if qbase is aligned to rbase*/ \
		h[m] = max(h[m], f[m]); \
		h[m] = max(h[m], 0); \
		e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);/*whether to introduce or extend a gap in target_batch sequence */\
		h[m] = max(h[m], e); \
		max_ref_idx = (max_score < h[m]) ? ref_idx + (m-1) : max_ref_idx; \
		max_score = (max_score < h[m]) ? h[m] : max_score; \
		p[m] = h[m-1];

#define CORE_COMPUTE() \
		uint32_t rbase = (packed_ref_literal >> l) & 15;\
		DEV_GET_SUB_SCORE_GLOBAL(temp_score, qbase, rbase) \
		temp_score += p[m]; \
		h[m] = max(temp_score, f[m]); \
		h[m] = max(h[m], e); \
		f[m] = max(temp_score- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(temp_score- _cudaGapOE, e - _cudaGapExtend); \
		p[m] = h[m-1]; \
		diag_idx = ((ref_idx + m-1+query_idx)&(total_shm-1))<<5;\
		antidiag_max[real_warp_id+diag_idx] = max(antidiag_max[real_warp_id+diag_idx], (h[m]<<16) +ref_idx+ m-1);\

#define CORE_COMPUTE_BOUNDARY() \
		if (query_idx + _cudaBandWidth < ref_idx + m-1 || query_idx - _cudaBandWidth > ref_idx + m-1) { \
			p[m] = h[m-1]; \
		} else { \
			uint32_t rbase = (packed_ref_literal >> l) & 15;\
			DEV_GET_SUB_SCORE_GLOBAL(temp_score, qbase, rbase) \
			temp_score += p[m]; \
			h[m] = max(temp_score, f[m]); \
			h[m] = max(h[m], e); \
			f[m] = max(temp_score- _cudaGapOE, f[m] - _cudaGapExtend); \
			e = max(temp_score- _cudaGapOE, e - _cudaGapExtend); \
			p[m] = h[m-1]; \
			diag_idx = ((ref_idx + m-1+query_idx)&(total_shm-1))<<5;\
			antidiag_max[real_warp_id+diag_idx] = max(antidiag_max[real_warp_id+diag_idx], (h[m]<<16) +ref_idx+ m-1);\
		}
		

__global__ void agatha_kernel(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint4 *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{
    /*Initial kernel setup*/

	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;

	// Initializing max score and its idx
    int32_t max_score = 0; 
	int32_t max_ref_idx = 0; 
    int32_t prev_max_score = 0;
    int32_t max_query_idx = 0;

	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses
	
	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* antidiag_max = (int32_t*)(shared_maxHH+(threadIdx.x/32)*total_shm*32);
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {
		
		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? global_ub_idx[n_tasks-(job_idx>>2)-1].y: global_ub_idx[job_idx-(job_idx>>2)-1].y;
				
		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= _cudaBandWidth? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= _cudaBandWidth? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= _cudaBandWidth? k: MINUS_INF2; 	
			}
		}
		
		// fill shared memory with initial value
		for (m = 0; m < total_shm; m++) {
			antidiag_max[real_warp_id + m*32] = INT_MIN;
		}

		__syncwarp();

		// Initialize variables
		max_score = 0; 
		prev_max_score = 0;
		max_ref_idx = 0; 
    	max_query_idx = 0;
		terminated = false;

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk

		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			
			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - _cudaBandWidth)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + _cudaBandWidth)/2/packed_len);
			finished_blocks = slice_start;
			
			if (slice_start > slice_end) {
				terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not
				
				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];

					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - _cudaBandWidth)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + _cudaBandWidth )) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}
					
				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;
						
						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence
							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;

							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_BOUNDARY();
								}
							} else {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE();
								}
							}
							
							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							query_idx++;

						}

					}
					

					packed_query_idx++;
					
				}
				
				// write intermediate values to global buffers
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( ref_idx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
						}
					}
					global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
				}
				
				finished_blocks+=warp_len;
			}

			__syncwarp();

			last_diag = (i+_cudaSliceWidth)<<3;
			prev_max_score = query_len+ref_len-1;

			/* Termination Condition & Score Update */
			if (!terminated) {
				for (diag_idx = i<<3; diag_idx < last_diag; diag_idx++) {
					if (diag_idx <prev_max_score) {
						m = diag_idx&(total_shm-1);
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = diag_idx-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (diag_idx-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (diag_idx-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								terminated = true;
								break;
							}
						}
						// reset shared memory buffer for next slice
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
			}
			
			__syncwarp();

			// If job is finished
			if (terminated) {
				total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
				if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
			}
			
			// Update the max score and its index to shared memory (used in Subwarp Rejoining)
			if (warp_id==1) shared_job[20+(warp_num&3)] = max_score;
			else if (warp_id==2) shared_job[24+(warp_num&3)] = (max_ref_idx<<16) + max_query_idx;
 
			__syncwarp();

			i += _cudaSliceWidth;

			/*Job wrap-up*/
			// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
			if (i >= total_anti_diags) {
				
				// In the case of (2), check the termination condition & score update for the last diagonal block
				if (!terminated) {
					diag_idx = (i*packed_len)&(total_shm-1);
					for (k = i*packed_len, m = diag_idx; m < diag_idx+packed_len; m++, k++) {
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = k-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (k-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (k-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								terminated = true;
								break;
							}
						}
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
				
				// Spill the results to GPU memory to be later moved to the CPU
				if (warp_id==0) {
					device_res->aln_score[ub_idx] = max_score;//copy the max score to the output array in the GPU mem
					device_res->query_batch_end[ub_idx] = max_query_idx;//copy the end position on query_batch sequence to the output array in the GPU mem
					device_res->target_batch_end[ub_idx] = max_ref_idx;//copy the end position on target_batch sequence to the output array in the GPU mem
				}

				/*Subwarp Rejoining*/
				// The subwarp that has no job looks for new jobs by iterating over other subwarp's job
				for (m = 0; m < (32/const_warp_len); m++) {
					// if the selected job still has remainig diagonals
					if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
						// read the selected job's info
						total_anti_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						ub_idx = shared_job[16+m];

						packed_ref_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						query_len = ref_len&65535;
						ref_len = ref_len>>16;
						packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
						packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);
						
						max_score = shared_job[20+m];
						max_ref_idx = shared_job[24+m];
						max_query_idx = max_ref_idx&65535;
						max_ref_idx = max_ref_idx>>16;
						
						// reset the flag
						terminated = false;

						// reset shared memory buffer
						for (m = 0; m < total_shm; m++) {
							antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
						}
						
						break;
					}
				}

			}

			__syncwarp();
			
			/*Subwarp Rejoining*/
			//Set the mask, warp length and thread id within the warp 
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;
			
			__syncwarp();

		}

		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];

		__syncwarp();



	}
	
	return;


}


__global__ void agatha_sort(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID

	uint32_t query_len, ref_len, packed_query_len, packed_ref_len;

	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	if (tid < n_tasks) {

		query_len = query_batch_lens[tid];
		ref_len = target_batch_lens[tid];
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);

		global_ub_idx[tid] = make_short2((packed_ref_len + packed_query_len-1), tid);


	}
	
	return;


}
#endif
