#include "gasal.h"

#include "args_parser.h"

#include "res.h"


gasal_res_t *gasal_res_new_host(uint32_t max_n_alns, Parameters *params)
{
	cudaError_t err;
	gasal_res_t *res = NULL;


	res = (gasal_res_t *)malloc(sizeof(gasal_res_t));

	CHECKCUDAERROR(cudaHostAlloc(&(res->aln_score), max_n_alns * sizeof(int32_t),cudaHostAllocDefault));
	
	
	if(res ==NULL)
	{
		fprintf(stderr,  "Malloc error on res host ");
		exit(1);
	}


	if (params->algo == GLOBAL) {
		res->query_batch_start = NULL;
		res->target_batch_start = NULL;
		res->query_batch_end = NULL;
		res->target_batch_end = NULL;
		/*
		// Deprecated. For semi-global you now need to know the start and stop positions.
		} else if (params->algo == SEMI_GLOBAL) {
		res->host_query_batch_start = NULL;
		res->host_query_batch_end = NULL;
		res->query_batch_start = NULL;
		res->query_batch_end = NULL;

		if (params->start_pos == WITH_START) {
		CHECKCUDAERROR(cudaHostAlloc(&(res->host_target_batch_start),max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaHostAlloc(&(res->host_target_batch_end),max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(res->target_batch_start),max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(
		cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
		} else {
		CHECKCUDAERROR(cudaHostAlloc(&(res->host_target_batch_end),max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
		res->host_target_batch_start = NULL;
		res->target_batch_start = NULL;
		}
		 */
	} else {
		if (params->start_pos == WITH_START || params->start_pos == WITH_TB) {
			CHECKCUDAERROR(cudaHostAlloc(&(res->query_batch_start),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->target_batch_start),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));

		} else {
			CHECKCUDAERROR(cudaHostAlloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			res->query_batch_start = NULL;
			res->target_batch_start = NULL;
		}

	}
	if (params->start_pos == WITH_TB) {
		CHECKCUDAERROR(cudaHostAlloc(&(res->n_cigar_ops), max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
	}

	return res;
}


gasal_res_t *gasal_res_new_device(gasal_res_t *device_cpy)
{
	cudaError_t err;


	
    // create class storage on device and copy top level class
    gasal_res_t *d_c;
    CHECKCUDAERROR(cudaMalloc((void **)&d_c, sizeof(gasal_res_t)));
	//    CHECKCUDAERROR(cudaMemcpy(d_c, res, sizeof(gasal_res_t), cudaMemcpyHostToDevice));



    // copy pointer to allocated device storage to device class
    CHECKCUDAERROR(cudaMemcpy(&(d_c->aln_score), &(device_cpy->aln_score), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->query_batch_start), &(device_cpy->query_batch_start), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->target_batch_start), &(device_cpy->target_batch_start), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->query_batch_end), &(device_cpy->query_batch_end), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->target_batch_end), &(device_cpy->target_batch_end), sizeof(int32_t*), cudaMemcpyHostToDevice));





	return d_c;
}




gasal_res_t *gasal_res_new_device_cpy(uint32_t max_n_alns, Parameters *params)
{
	cudaError_t err;
	gasal_res_t *res;

	res = (gasal_res_t *)malloc(sizeof(gasal_res_t));

	CHECKCUDAERROR(cudaMalloc(&(res->aln_score), max_n_alns * sizeof(int32_t)));

	if (params->algo == GLOBAL) {
		res->query_batch_start = NULL;
		res->target_batch_start = NULL;
		res->query_batch_end = NULL;
		res->target_batch_end = NULL;

	} else {
		if (params->start_pos == WITH_START || params->start_pos == WITH_TB) {

			CHECKCUDAERROR(cudaMalloc(&(res->query_batch_start),max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->target_batch_start),max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
		
		} else {
		
			CHECKCUDAERROR(cudaMalloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
		
			res->query_batch_start = NULL;
			res->target_batch_start = NULL;
		}

	}
	return res;
}

// TODO : make 2 destroys for host and device
void gasal_res_destroy_host(gasal_res_t *res) 
{
	cudaError_t err;
	if (res == NULL)
		return;


	if (res->aln_score != NULL) CHECKCUDAERROR(cudaFreeHost(res->aln_score));
	if (res->query_batch_start != NULL) CHECKCUDAERROR(cudaFreeHost(res->query_batch_start));
	if (res->target_batch_start != NULL) CHECKCUDAERROR(cudaFreeHost(res->target_batch_start));
	if (res->query_batch_end != NULL) CHECKCUDAERROR(cudaFreeHost(res->query_batch_end));
	if (res->target_batch_end != NULL) CHECKCUDAERROR(cudaFreeHost(res->target_batch_end));
	//if (res->n_cigar_ops != NULL) CHECKCUDAERROR(cudaFreeHost(res->n_cigar_ops));

	free(res);
}

void gasal_res_destroy_device(gasal_res_t *device_res, gasal_res_t *device_cpy) 
{
	cudaError_t err;
	if (device_cpy == NULL || device_res == NULL)
		return;

	if (device_cpy->aln_score != NULL) CHECKCUDAERROR(cudaFree(device_cpy->aln_score));
	if (device_cpy->query_batch_start != NULL) CHECKCUDAERROR(cudaFree(device_cpy->query_batch_start));
	if (device_cpy->target_batch_start != NULL) CHECKCUDAERROR(cudaFree(device_cpy->target_batch_start));
	if (device_cpy->query_batch_end != NULL) CHECKCUDAERROR(cudaFree(device_cpy->query_batch_end));
	if (device_cpy->target_batch_end != NULL) CHECKCUDAERROR(cudaFree(device_cpy->target_batch_end));
	//if (device_cpy->cigar != NULL) CHECKCUDAERROR(cudaFree(device_cpy->cigar));
	

	CHECKCUDAERROR(cudaFree(device_res));
	
	free(device_cpy);
}
