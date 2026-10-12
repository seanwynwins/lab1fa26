#include <unistd.h>
#include "listutils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

void parallelListRanks (long head, const long* next, long* rank, size_t n)
{
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

	for (int iter = 0; iter < (int)ceil(log2(n)); iter++) {
		memcpy(neighbor_copy, neighbor, n * sizeof(long));
		memcpy(rank_copy, rank, n * sizeof(long));
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			if (neighbor_copy[i] != NIL) {
				rank[i] = rank_copy[i] + rank_copy[neighbor_copy[i]];
				neighbor[i] = neighbor_copy[neighbor_copy[i]];
			}
		}

	}
	
	long inverse = rank[head];

	for (int i = 0; i < n; i++) {
		rank[i] = inverse - rank[i];
	}
	
	free(neighbor);
	free(neighbor_copy);
	free(rank_copy);
}