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
		if (compact_col + W < compact_row + m-1 || compact_col - W > compact_row + m-1) continue;\
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
		
#define CORE_LOCAL_COMPUTE_ORIGINAL() \
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


// Ver 0.
// if (read_iter*packed_len + W + (28-k)/4 < (j*warp_len+warp_id)*packed_len + m-1 || read_iter*packed_len - W + (28-k)/4 > (j*warp_len + warp_id)*packed_len + m-1) continue;\



/* typename meaning : 
    - T is the algorithm type (LOCAL, MICROLOCAL)
    - S is WITH_ or WIHTOUT_START
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/
template <typename T, typename S, typename B>
__global__ void gasal_local_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint4 *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_inter_row)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	//if (tid >= n_tasks) return;

	int32_t i, j, k, m, l;
	int32_t e;

    int32_t maxHH = 0; //initialize the maximum score to zero
	int32_t maxXY_y = 0; 

    int32_t prev_maxHH = 0;
    int32_t maxXY_x = 0;


	int32_t subScore;

	int32_t gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);
	
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

	uint32_t packed_target_batch_idx, packed_query_batch_idx, read_len, ref_len, query_batch_regs, target_batch_regs;

	const int packed_len = 8;
	//const int shared_len = 32;
	//int iter_num;
	int W = 500;
	int band_start, band_end;
	int row_start, row_end;
	int diag;
	int read_iter;
	register uint32_t gpac, rpac;
	int compact_row, compact_col; 
	bool diff_row = false;
	bool cont_flag;
	int bound;


	for (i = job_start_idx; i < job_end_idx; i++) {

		maxHH = 0; //initialize the maximum score to zero
		maxXY_y = 0; 

    	prev_maxHH = 0;
    	maxXY_x = 0;
		
		// get target and query seq
		packed_target_batch_idx = target_batch_offsets[i] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[i] >> 3;//starting index of the query_batch sequence
		read_len = query_batch_lens[i];
		ref_len = target_batch_lens[i];
		query_batch_regs = (read_len >> 3) + (read_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		target_batch_regs = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// fill with initial value
		for (j = 0; j < job_per_query; j++) {
			if ((j*warp_len + warp_id) < max_query_len) {
				global_inter_row[warp_num*max_query_len + j*warp_len + warp_id] = initHD;	
				//inter_row[warp_block_id*512 + j*warp_len + warp_id] = global_inter_row[warp_num*max_query_len + j*warp_len + warp_id];	
			}
		}

		//initialize values
		j = 0;
		
		//for the leader thread
		band_start = 0;
		band_end = ( min(query_batch_regs-1, ( ( (j*warp_len)*packed_len + packed_len -1 + W ) + packed_len -1 )/packed_len) );

		//for each individual thread
		compact_row = warp_id;
		if (compact_row >= target_batch_regs) {
			cont_flag = false;
		} else {
			cont_flag = true;
		}
		// read packed elements from ref
		gpac = packed_target_batch[packed_target_batch_idx + warp_id];
		gidx = (warp_id) << 3;

		//initialize band values 
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}

		read_iter = -(warp_id);

		row_start = ((warp_id*packed_len - W) < 0) ? ((warp_id*packed_len - W)-packed_len+1)/packed_len : ((warp_id*packed_len - W)+packed_len-1)/packed_len;
		row_start += warp_id;
		row_end = compact_row + ( min( query_batch_regs-1, ( ( compact_row*packed_len + packed_len -1 + W ) + packed_len -1) /packed_len ) );
		compact_row *= packed_len;

		diag = max(warp_id, row_start);
		diff_row = false; 

		//BOUND
		/*
		bound = 0;
		*/
		// loop for row-chunks that have row_start < 0
		while(cont_flag) {

			// keep track of the first thread diag in each warp	
			// start of a new row at the first thread
			// band_start: where warp_id 0's diag is 
			// band_end: where warp_id 0's row ends 
			// diff_row: if true, current thread is on a different row chunk with warp_id 0
			// j: current idx of the row chunk 
			if (band_start > band_end) {
				j++;
				//band_start = (j*warp_len) + ( (max(0, (j*warp_len*packed_len - W)) + packed_len -1)/packed_len );
				band_start = (j*warp_len*packed_len - W);
				band_start = (band_start < 0)? (band_start-packed_len+1)/packed_len : (band_start+packed_len-1)/packed_len;
				band_start += j*warp_len;
				band_end = (j*warp_len) + ( min(query_batch_regs-1, ( ( (j*warp_len)*packed_len + packed_len -1 + W ) + packed_len -1 )/packed_len) );
				//first_diag = band_start;
				diff_row = true;
				//BOUND
				/*
				bound += warp_len;
				
				*/
				//check if this new row-chunk's last thread doesn't have a negative row_start
				// if true, go on to the next while statement 
				if (band_start >=0) break;
			}

			// diag: which diagonal the current thread is processing
			// row_start: the current thread's starting diagonal 
			// row_end: the current thread's ending diagonal 
			// start of a new row
			if (diag > row_end) {
				compact_row = j*warp_len + warp_id;
				// remove each thread that is out of ref bound
				if (compact_row >= target_batch_regs) {
					cont_flag = false;
					break;
				}

				// read packed elements from ref
				gpac = packed_target_batch[packed_target_batch_idx + j*warp_len + warp_id];
				gidx = (j*warp_len + warp_id) << 3;

				//initialize band values 
				for (m = 0; m < 9; m++) {
					h[m] = 0;
					f[m] = 0;
					p[m] = 0;
				}

				// set read_iter (starting block position of each thread) (thread-wise)
				//read_iter = (max(0, (compact_row*packed_len - W)) + packed_len -1)/packed_len;
				read_iter = ((compact_row*packed_len - W) < 0) ? ((compact_row*packed_len - W)-packed_len+1)/packed_len : ((compact_row*packed_len - W)+packed_len-1)/packed_len;
				
				// get start and end diag for each row (thread-wise) (in units of blocks)
				//row_start = (j*warp_len + warp_id) + ( read_iter );	// block ref idx + block query idx
				row_start = compact_row + ( read_iter );	// block ref idx + block query idx
				row_end = compact_row + ( min( query_batch_regs-1, ( ( compact_row*packed_len + packed_len -1 + W ) + packed_len -1) /packed_len ) ); 

				compact_row *= packed_len;

				diag = max(row_start, j*warp_len+warp_id);
				diff_row = false;

			}

			// if current diag within bound

			if ((read_iter >= 0)&&(diag <= band_start || diff_row)) {
				// read packed elements from query 
				rpac = packed_query_batch[packed_query_batch_idx + read_iter]; 
				//ridx = 0;
				
				compact_col = read_iter*packed_len;
				// calculate scores for the current block
				//for (k = 28; k >= 0 && (ridx+read_iter*8) < read_len; k -= 4) {
				
				for (k = 28; k >= 0 && compact_col < read_len; k -= 4) {
					uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
					//-----load intermediate values--------------
					//HD = inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx];
					//HD = global_inter_row[warp_num*max_query_len + read_iter*packed_len + ridx];
					HD = global_inter_row[warp_num*max_query_len + compact_col];
					h[0] = HD.x;
					e = HD.y;

					if (diag < row_start + 3 || row_end - 3 < diag) {
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
					global_inter_row[warp_num*max_query_len + compact_col] = HD;
					//---------------------------------------------


					//maxXY_x = (prev_maxHH < maxHH) ? ridx+read_iter*8 : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
					maxXY_x = (prev_maxHH < maxHH) ? compact_col : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
					prev_maxHH = max(maxHH, prev_maxHH);
					//ridx++;
					compact_col++;
					//-------------------------------------------------------

				}
				

				// update read_iter
				
				diag++;
				read_iter++;
				
			}

			if (read_iter < 0) read_iter++;
			band_start++;
			/*
			//BOUND
			if (bound == band_start) {

			}
			*/
		
		}


		
		// loop for row-chunks that have row_start >= 0
		while(cont_flag) {

			// keep track of the first thread diag in each warp	
			// start of a new row at the first thread
			// band_start: where warp_id 0's diag is 
			// band_end: where warp_id 0's row ends 
			// diff_row: if true, current thread is on a different row chunk with warp_id 0
			// j: current idx of the row chunk 
			if (band_start > band_end) {
				j++;
				band_start = (j*warp_len) + ( (max(0, (j*warp_len*packed_len - W)) + packed_len -1)/packed_len );
				band_end = (j*warp_len) + ( min(query_batch_regs-1, ( ( (j*warp_len)*packed_len + packed_len -1 + W ) + packed_len -1 )/packed_len) );
				//first_diag = band_start;
				diff_row = true;
				/*
				//BOUND
				bound += warp_len;
				*/
				
			}

			// diag: which diagonal the current thread is processing
			// row_start: the current thread's starting diagonal 
			// row_end: the current thread's ending diagonal 
			// start of a new row
			if (diag > row_end) {
				compact_row = j*warp_len + warp_id;
				// remove each thread that is out of ref bound
				if (compact_row >= target_batch_regs) {
					cont_flag = false;
					break;
				}

				// read packed elements from ref
				gpac = packed_target_batch[packed_target_batch_idx + j*warp_len + warp_id];
				gidx = (j*warp_len + warp_id) << 3;

				//initialize band values 
				for (m = 0; m < 9; m++) {
					h[m] = 0;
					f[m] = 0;
					p[m] = 0;
				}

				// set read_iter (starting block position of each thread) (thread-wise)
				read_iter = (max(0, (compact_row*packed_len - W)) + packed_len -1)/packed_len;
				
				// get start and end diag for each row (thread-wise) (in units of blocks)
				//row_start = (j*warp_len + warp_id) + ( read_iter );	// block ref idx + block query idx
				row_start = compact_row + ( read_iter );	// block ref idx + block query idx
				//row_end = (j*warp_len + warp_id) + ( (min(read_len-1, ( (j*warp_len + warp_id)*packed_len + packed_len -1 + W )) + packed_len -1)/packed_len ); 
				//row_end = (j*warp_len + warp_id) + ( min( query_batch_regs-1, ( ( (j*warp_len + warp_id)*packed_len + packed_len -1 + W ) + packed_len -1) /packed_len ) ); 
				row_end = compact_row + ( min( query_batch_regs-1, ( ( compact_row*packed_len + packed_len -1 + W ) + packed_len -1) /packed_len ) ); 

				compact_row *= packed_len;

				diag = row_start;
				diff_row = false;

			}

			// if current diag within bound

			if (diag <= band_start || diff_row) {
				// read packed elements from query 
				rpac = packed_query_batch[packed_query_batch_idx + read_iter]; 
				//ridx = 0;
				
				compact_col = read_iter*packed_len;
				// calculate scores for the current block
				//for (k = 28; k >= 0 && (ridx+read_iter*8) < read_len; k -= 4) {
				
				for (k = 28; k >= 0 && compact_col < read_len; k -= 4) {
					uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
					//-----load intermediate values--------------
					//HD = inter_row[warp_block_id*packed_warp_len + (read_iter%(2*shared_len))*packed_len + ridx];
					//HD = global_inter_row[warp_num*max_query_len + read_iter*packed_len + ridx];
					HD = global_inter_row[warp_num*max_query_len + compact_col];
					h[0] = HD.x;
					e = HD.y;

					if (diag < row_start + 3 || row_end - 3 < diag) {
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
					global_inter_row[warp_num*max_query_len + compact_col] = HD;
					//---------------------------------------------


					//maxXY_x = (prev_maxHH < maxHH) ? ridx+read_iter*8 : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
					maxXY_x = (prev_maxHH < maxHH) ? compact_col : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
					prev_maxHH = max(maxHH, prev_maxHH);
					//ridx++;
					compact_col++;
					//-------------------------------------------------------

				}
				

				// update read_iter
				read_iter++;
				diag++;
				
			}
			
			band_start++;
			/*
			//BOUND
			if (bound == band_start) {
				
			}
			*/

		
		}



		__syncwarp();
		
		// add max values to shared memory 
	
		//reduction on max value
		shared_maxHH[tx] = maxHH;
		shared_maxHH[blockDim.x + tx] = maxXY_x;
		shared_maxHH[blockDim.x*2 + tx] = maxXY_y;


		for (int y=2; y <= warp_len; y*=2) {
			if (warp_id%y == 0 && warp_id < (warp_len - y/2)  ) {
				if (shared_maxHH[tx] < shared_maxHH[tx+y/2]  || (shared_maxHH[tx]==shared_maxHH[tx+y/2]&& shared_maxHH[blockDim.x*2+tx] > shared_maxHH[blockDim.x*2+tx+y/2])) {
					shared_maxHH[tx] = shared_maxHH[tx+y/2];
					shared_maxHH[blockDim.x + tx] = shared_maxHH[blockDim.x + tx+y/2];
					shared_maxHH[blockDim.x*2+tx] = shared_maxHH[blockDim.x*2+tx+y/2];
				}
			}
		}
		
		if (warp_id==0) {
			device_res->aln_score[i] = shared_maxHH[tx];//copy the max score to the output array in the GPU mem
			device_res->query_batch_end[i] = shared_maxHH[blockDim.x+tx];//copy the end position on query_batch sequence to the output array in the GPU mem
			device_res->target_batch_end[i] = shared_maxHH[blockDim.x*2+tx];//copy the end position on target_batch sequence to the output array in the GPU mem
		}


	}
	



	///////////////////////////////////////////////

	



	return;


}
#endif
