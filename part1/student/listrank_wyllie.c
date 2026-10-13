#include <unistd.h>
#include "listutils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

void parallelListRanks (long head, const long* next, long* rank, size_t n)
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		rank[i] = 1;
		if (next[i] == NIL) {
			rank[i] = 0;	
		}
	}

	long *neighbor = malloc(n * sizeof(long));
	long *neighbor_copy = malloc(n * sizeof(long));
	long *rank_copy = malloc(n * sizeof(long));

	memcpy(neighbor, next, n * sizeof(long));

	// try using double buffering?
	// https://www.geeksforgeeks.org/dbms/double-buffering/
	long *rank_current = rank;
	long *neighbor_current = neighbor;
	long *rank_next = rank_copy;
	long *neighbor_next = neighbor_copy;

	for (int iter = 0; iter < (int)ceil(log2(n)); iter++) {
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			if (neighbor_current[i] != NIL) {
				rank_next[i] = rank_current[i] + rank_current[neighbor_current[i]];
				neighbor_next[i] = neighbor_current[neighbor_current[i]];
			} else {
				rank_next[i] = rank_current[i];
				neighbor_next[i] = NIL;
			}
		}
		
		long *tmp = rank_current;
		rank_current = rank_next;
		rank_next = tmp;
		
		tmp = neighbor_current;
		neighbor_current = neighbor_next;
		neighbor_next = tmp;
	}
	
	if (rank_current != rank) {
		memcpy(rank, rank_current, n * sizeof(long));
	}
	
	long inverse = rank[head];

	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		rank[i] = inverse - rank[i];
	}
	
	free(neighbor);
	free(neighbor_copy);
	free(rank_copy);
}