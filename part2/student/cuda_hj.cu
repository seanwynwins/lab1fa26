#include <unistd.h>
#include "listutils.h"
#include <unistd.h>
#include <omp.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

// using this ref: https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// just move pragma stuff to global gfunction
// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// 
/*
The migration is expensive because page faults occur individually, and GPU threads stall while they wait for the page migration. Because I know what memory is needed by the kernel (x and y arrays), I can use prefetching to make sure that the data is on the GPU before the kernel needs it. I do this by using the cudaMemPrefetchAsync() function before launching the kernel:

// Prefetch the x and y arrays to the GPU
cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

*/

int blockSize = 256;

__global__
void sublist_ranking(long *cuda_next, long *cuda_rank, bool *is_head, 
long *sublist_id, long *sublist_length, long *sublist_next_node, long n,
long *cuda_head) {
	int headIndex  = blockIdx.x * blockDim.x + threadIdx.x;
	if (headIndex >= n) {
		return;
	}
	int i = cuda_head[headIndex];
	long current_rank = 0;
	long current_node = i;
				// make sure the node's not the head
		do {
			// set list ranking for sublist on processor
			cuda_rank[current_node] = current_rank;
			sublist_id[current_node] = i;
			current_rank++;
			current_node = cuda_next[current_node];
		}
		while (current_node != NIL && !is_head[current_node]);
		sublist_length[i] = current_rank;
		sublist_next_node[i] = current_node;
}

__global__
void offset(long *cuda_rank, long *rank_offset, long *sublist_id, long n) {
	// rank offsets to all nodes in the sublists
	// #pragma omp parallel for
	// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	cuda_rank[i] += rank_offset[sublist_id[i]];
}	

extern "C" void parallelListRanks(const long head, const long* next, long* rank, const size_t n)
{
	// First, I just have to turn our add function into a function that the GPU can run, called a kernel in CUDA. 
	// To allocate data in unified memory, call cudaMallocManaged(), which returns a pointer that you can access from host (CPU) code or device (GPU) code. To free the data, just pass the pointer to cudaFree().
	// CUDA kernel launches are specified using the triple angle bracket syntax <<< >>>
	// first rgaument is block, blocks are logical groupings of threads
	// There are two parameters here, but let’s start by changing the second one: the number of threads in a thread block.
	// reference : https://downey.io/blog/helman-jaja-list-ranking-explained/

	// random heads
	// Split the linked list into sublists by randomly choosing at least P - 1 sublist heads in addition to the list’s true head to create s sublists
	// int num_processes = omp_get_max_threads();

	// ref: So, in practice, I actually saw improved performance by choosing a value of s that was greater than P and allowing OpenMP to sort of load balance the work across the processors.
	/// just pick some big number
	// adjust this...
	int sublists = n / blockSize;
	if (sublists < 1) {
		sublists = 1;
	} else if (sublists > n) {
		sublists = n;
	}
	// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
	int numSublistBlocks = (sublists + blockSize - 1) / blockSize;
	int numBlocks = (n + blockSize - 1) / blockSize;

	// keep trrack of bunch of sublist information
	bool *is_head;
	long *sublist_id;
	long *sublist_length;
	long *sublist_next_node;
	long *rank_offset;
	long *cuda_next;
	long *cuda_rank;
	long *cuda_head;
	
	cudaMallocManaged(&cuda_next, n*sizeof(long));
	cudaMallocManaged(&cuda_rank, n*sizeof(long));
	cudaMallocManaged(&is_head, n*sizeof(bool));
	cudaMallocManaged(&sublist_id, n*sizeof(long));
	cudaMallocManaged(&sublist_length, n*sizeof(long));
	cudaMallocManaged(&sublist_next_node, n*sizeof(long));
	cudaMallocManaged(&rank_offset, n*sizeof(long));
	cudaMallocManaged(&cuda_head, sublists*sizeof(long));
	memcpy(cuda_next, next, n * sizeof(long));


	for (int i = 0; i < n; i++) {
		is_head[i] = false;
	}

	cuda_head[0] = head;
	is_head[head] = true;	

	// random choose sublists - 1 sublist heads
	for (int i = 1; i < sublists; i++) {
		long node = rand() % n;
	// account for duplicate somehow...or maybe it doesnt really matter that much
	// maybe this will help speed up so try it in cuda
	while(is_head[node]) {
		node = rand() % n;
	}
		is_head[node] = true;
		cuda_head[i] = node;
	}
	
	cudaMemPrefetchAsync(is_head, n*sizeof(bool), 0, 0);
	cudaMemPrefetchAsync(sublist_id, n*sizeof(long), 0, 0);
	cudaMemPrefetchAsync(sublist_length, n*sizeof(long), 0, 0);
	cudaMemPrefetchAsync(sublist_next_node, n*sizeof(long), 0, 0);
	cudaMemPrefetchAsync(rank_offset, n*sizeof(long), 0, 0);
	cudaMemPrefetchAsync(cuda_next, n*sizeof(long), 0, 0);
	cudaMemPrefetchAsync(cuda_rank, n*sizeof(long), 0, 0);
	cudaMemPrefetchAsync(cuda_head, sublists*sizeof(long), 0, 0);
	sublist_ranking<<<numSublistBlocks, blockSize>>>(cuda_next, cuda_rank, is_head, sublist_id, sublist_length, sublist_next_node, sublists, cuda_head);
	cudaDeviceSynchronize();

	// head should actually be beginning
	
	rank_offset[head] = 0;
	long previous_node = head;
	long current_node = sublist_next_node[head];

	// we can walk across the sublist heads themselves in the correct order since we recorded each sublist’s successor index.
	while (current_node != NIL) {
		rank_offset[current_node] = rank_offset[previous_node] + sublist_length[previous_node];
		previous_node = current_node;
		current_node = sublist_next_node[current_node];
	}

	cudaMemPrefetchAsync(rank_offset, n*sizeof(long), 0, 0);
	offset<<<numBlocks, blockSize>>>(cuda_rank, rank_offset, sublist_id, n);
	cudaDeviceSynchronize();

	memcpy(rank, cuda_rank, n * sizeof(long));
	cudaFree(is_head);
	cudaFree(sublist_id);
	cudaFree(sublist_length);
	cudaFree(sublist_next_node);
	cudaFree(rank_offset);
	cudaFree(cuda_next);
	cudaFree(cuda_rank);
	cudaFree(cuda_head);
}
