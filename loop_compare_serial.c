#include <stdio.h>
#include <math.h>
#include <sys/time.h>


// A simple helper function to compute the difference between to points in time
double time_diff(struct timeval x , struct timeval y){
	double x_ms , y_ms , diff;
	 
	x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
	y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;
	 
	diff = (double)y_ms - (double)x_ms;
	 
	return diff;
}

// A simple helper function to reset two arrays with random values


int main(int argc, char ** argv) {

	struct timeval beforeA, afterA, beforeB, afterB;

	double sumA = 0.0, avgTimeA = 0.0, minTimeA = INFINITY, maxTimeA = 0.0, sumB = 0.0, avgTimeB = 0.0, minTimeB = INFINITY, maxTimeB = 0.0;

	int numTrials = 100000;

	for (int trial = 0; trial < numTrials; trial++){

		// time the Parallel tiles
		gettimeofday(&beforeA , NULL);
		float aTest;
		for (int i = 0; i < 10000; i++){
			aTest = i*2;
		}
		gettimeofday(&afterA , NULL);

		// time the Serial tiles
		gettimeofday(&beforeB, NULL);
		for (int i = 0; i < 10000; i++){
			float bTest = i*2;
		}
		gettimeofday(&afterB, NULL);

		// compute time comparison
		double timeTakenA = time_diff(beforeA , afterA);
		sumA += timeTakenA;

		if (timeTakenA < minTimeA){
			minTimeA = timeTakenA;
		}
		if (timeTakenA > maxTimeA){
			maxTimeA = timeTakenA;
		}

		// compute time comparison
		double timeTakenB = time_diff(beforeB , afterB);
		sumB += timeTakenB;

		if (timeTakenB < minTimeB){
			minTimeB = timeTakenB;
		}
		if (timeTakenB > maxTimeB){
			maxTimeB = timeTakenB;
		}

	} // trialsloop

	// compute the average time for each scenario
	avgTimeA= sumA/numTrials;
	avgTimeB = sumB/numTrials;
	printf("\tA\n\t\tAvg time : %.0lf us\tMin Time : %.0lf us\tMax Time : %.0lf us\n\tB\n\t\tAvg time : %.0lf us\tMin Time : %.0lf us\tMax Time : %.0lf us\n\n", avgTimeA, minTimeA, maxTimeA, avgTimeB, minTimeB, maxTimeB); 

	return 0;
}