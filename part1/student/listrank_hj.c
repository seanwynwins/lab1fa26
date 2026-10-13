#include <unistd.h>
#include "listutils.h"
#include <omp.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

void parallelListRanks (long head, const long* next, long* rank, size_t n) {

	// reference : https://downey.io/blog/helman-jaja-list-ranking-explained/

	// random heads
	// Split the linked list into sublists by randomly choosing at least P - 1 sublist heads in addition to the list’s true head to create s sublists
	int num_processes = omp_get_max_threads();

	// ref: So, in practice, I actually saw improved performance by choosing a value of s that was greater than P and allowing OpenMP to sort of load balance the work across the processors.
	int sublists = num_processes * 4;
	
	// keep trrack of bunch of sublist information
	bool *is_head = malloc(n * sizeof(bool));
	long *sublist_id = malloc(n * sizeof(long));
	long *sublist_length = malloc(n * sizeof(long));
	long *sublist_next_node = malloc(n * sizeof(long));
	long *rank_offset = malloc(n * sizeof(long));

	for (int i = 0; i< n; i++){
		sublist_id[i] = -1;
		sublist_length[i] = 0;
		sublist_next_node[i] = NIL;
		rank_offset[i] = 0;
	}

	for (int i = 0; i < n; i++) {
		is_head[i] = false;
	}

	is_head[head] = true;	

	// random choose sublists - 1 sublist heads
	for (int i = 0; i < sublists - 1; i++) {
		long node = rand() % n;
	// account for duplicate somehow...or maybe it doesnt really matter that much
		is_head[node] = true;
	}

	// traverse each subliststopping when reach head of another sublist
	#pragma omp parallel for
	for(int i = 0; i < n; i++) {
	if (is_head[i]) {
		long current_rank = 0;
		long current_node = i;
		while (current_node != NIL && (current_node == i || !is_head[current_node])) {
			// set list ranking for sublist on processor
			rank[current_node] = current_rank;
			sublist_id[current_node] = i;
			current_rank++;
			current_node = next[current_node];

		}
			
		// reached end of sublist - record total rank/index of next sublist head
		sublist_length[i] = current_rank;
		sublist_next_node[i] = current_node;

	}
	}	
	rank_offset[head] = 0;
	long previous_node = head;
	long current_node = sublist_next_node[head];

	// we can walk across the sublist heads themselves in the correct order since we recorded each sublist’s successor index.
	while (current_node != NIL) {
		rank_offset[current_node] = rank_offset[previous_node] + sublist_length[previous_node];
		previous_node = current_node;
		current_node = sublist_next_node[current_node];
	}
	// rank offsets to all nodes in the sublists
	#pragma omp parallel for
	for(int i = 0; i < n; i++) {
		rank[i] += rank_offset[sublist_id[i]];
	}

	free(is_head);
	free(sublist_id);
	free(sublist_length);
	free(sublist_next_node);
	free(rank_offset);
}
