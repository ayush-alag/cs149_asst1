#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "CycleTimer.h"

using namespace std;

// Timing variables for profiling
static double totalAssignmentsTime = 0.0;
static double totalCentroidsTime = 0.0;
static double totalCostTime = 0.0;

typedef struct {
  // Control work assignments
  int start_k, end_k;
  int start_m, end_m;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 *
 * @param prevCost Pointer to the K dimensional array containing cluster costs
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 *
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 *
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  double result = sqrt(accum);

  return result;
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  int chunkSize = args->end_m - args->start_m;
  double *minDist = new double[chunkSize];

  // Initialize arrays
  for (int m = 0; m < chunkSize; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[args->start_m + m] = -1;
  }

  // Assign datapoints to closest centroids
  for (int m = args->start_m; m < args->end_m; m++) {
    for (int k = args->start_k; k < args->end_k; k++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[m - args->start_m]) {
        minDist[m - args->start_m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  free(minDist);
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  free(counts);
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start_k; k < args->end_k; k++) {
    args->currCost[k] = accum[k];
  }

  free(accum);
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in
 *     the array. The N values of the i'th datapoint are the N values in the
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Reset timing variables
  totalAssignmentsTime = 0.0;
  totalCentroidsTime = 0.0;
  totalCostTime = 0.0;

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  static constexpr int MAX_THREADS = 32;

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  std::thread workers[MAX_THREADS];
  WorkerArgs args[MAX_THREADS];

  for (int i=0; i<MAX_THREADS; i++) {
      args[i].data = data;
      args[i].clusterCentroids = clusterCentroids;
      args[i].clusterAssignments = clusterAssignments;
      args[i].currCost = currCost;
      args[i].M = M;
      args[i].N = N;
      args[i].K = K;

      args[i].start_k = 0;
      args[i].end_k = K;
      args[i].start_m = i * (M / MAX_THREADS);
      args[i].end_m = (i == MAX_THREADS - 1) ? M : (i + 1) * (M / MAX_THREADS);
  }

  // Spawn the worker threads.  Note that only numThreads-1 std::threads
  // are created and the main application thread is used as a worker
  // as well.

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    double startAssignments = CycleTimer::currentSeconds();

    for (int i=1; i<MAX_THREADS; i++) {
        workers[i] = std::thread(computeAssignments, &args[i]);
    }

    computeAssignments(&args[0]);

    // join worker threads
    for (int i=1; i<MAX_THREADS; i++) {
        workers[i].join();
    }

    double endAssignments = CycleTimer::currentSeconds();
    totalAssignmentsTime += (endAssignments - startAssignments);

    double startCentroids = CycleTimer::currentSeconds();
    computeCentroids(&args[0]);
    double endCentroids = CycleTimer::currentSeconds();
    totalCentroidsTime += (endCentroids - startCentroids);

    double startCost = CycleTimer::currentSeconds();
    computeCost(&args[0]);
    double endCost = CycleTimer::currentSeconds();
    totalCostTime += (endCost - startCost);

    iter++;
  }

  // Print profiling results
  printf("\n=== Performance Profile ===\n");
  printf("Total time in computeAssignments(): %.4f ms (%.2f%%)\n",
         totalAssignmentsTime * 1000,
         100.0 * totalAssignmentsTime / (totalAssignmentsTime + totalCentroidsTime + totalCostTime));
  printf("Total time in computeCentroids():   %.4f ms (%.2f%%)\n",
         totalCentroidsTime * 1000,
         100.0 * totalCentroidsTime / (totalAssignmentsTime + totalCentroidsTime + totalCostTime));
  printf("Total time in computeCost():        %.4f ms (%.2f%%)\n",
         totalCostTime * 1000,
         100.0 * totalCostTime / (totalAssignmentsTime + totalCentroidsTime + totalCostTime));
  printf("Total algorithm time:               %.4f ms\n",
         (totalAssignmentsTime + totalCentroidsTime + totalCostTime) * 1000);
  printf("Number of iterations: %d\n", iter);
  printf("===========================\n\n");

  free(currCost);
  free(prevCost);
}
