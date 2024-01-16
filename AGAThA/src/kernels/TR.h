//mask: shows current idx within warp (should be set at the beginning as const)
const unsigned mask = (0xffffffff) << warp_id;
//job_array(shm): job list that are currently running
//job_thread_array(shm): number of more possible threads on each job 
//job_curr_thread_array(shm): number of current allocated threads on each job


//zdropped: flag for zdrop
//idle: shows whether a thread is idle or not
//job_start: the head of the job_array
//job_end: the tail of the job_array 
//TODO: job_end & job_start must be in shm as well 

if (zdropped) {
    //set current zdropped threads as idle 
    idle = 1;  
    //redist_flag(shm): set true if zdrop occurred 
    /*
    if (warp_id == 0) {
        redist_flag = true; 
    } 
    */ 
}

// check redist_flag(shm)
//if(redist_flag) {
// Or could use either warp match or warp vote function 
if (__any_sync(0xffffffff, zdropped)) {
    //if the current job was the only job in the job_array, get new jobs for the warp
    if (job_end == -1) {
        break;
        //TODO: this doesn't mean that all the jobs are over. 
        //means that all jobs have enough threads & no more TR is needed
        //need to somehow get to the start position
        //when arriving at the starting position, need to reset
        //zdropped, idle, job_start, job_end...
    }

    //start TR
    idle_threads = __ballot_sync(0xffffffff, idle);
    job_count = job_end-job_start+1;
    while(true) {
        
        if (idle) {
            //check the current idx within idling threads
            idle_id = __popc(mask & idle_threads);

            //try allocating to each job 
            warp_id = idle_id/job_count;
            job_idx = job_array[idle_id%job_count];
            if (warp_id < job_thread_array[job_idx]) {
                //the thread has been accepted to the job
                idle = 0;
                //TODO: update multiple init values including i
                
                warp_id += job_curr_thread_array[job_idx];                
            } 
        }

        //update job_...
        if (warp_id == 0) {
            //count current threads with the same job_idx
            job_curr_thread_array[job_idx] = __popc(__match_any_sync(0xffffffff, i));
            if (remaining_threads - job_curr_thread_array[job_idx] < job_thread_array[job_idx]) {
                job_thread_array[job_idx] = min(job_thread_array[job_idx], remaining_threads - job_curr_thread_array[job_idx]);
            }
            if (job_thread_array[job_idx] == 0) {
                // if job doesn't need more threads anymore
                //update job_end
                job_end = atomicSub(job_end, 1);
                //swap 
                job_array[job_idx] = job_array[job_end];
                job_thread_array[job_idx] = job_thread_array[job_idx];
                job_curr_thread_array[job_idx] = job_curr_thread_array[job_idx];
                
            }
        }        

        //check if more TR is needed
        idle_threads = __ballot_sync(0xffffffff, idle);
        if (idle_threads==0 | job_end == -1) {//if there are no threads to allocate | no job to allocate threads
            break;
        }
    
    }

    //set zdrop to false
    zdropped = false;     

}


//TODO: how to get new jobs when all the jobs are done
//TODO: how to check if all jobs are actually done