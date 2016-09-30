#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include "tiles3.c"
#include "parallelTiles.cu"


// HELPER FUNCTIONS
// Print an array of floats in [,,] format
void printFloatArray(float *arr, int len){
	printf("[");
	for (int i = 0; i < len -1; ++i) {
		printf("%.2f, ", arr[i]);
	}
	printf("%.2f]\n", arr[len-1]);
}

// Print an array of unsigned ints in [,,] format
void printUnsignedArray(unsigned *arr, int len){
	printf("[");
	for (int i = 0; i < len -1; ++i) {
		printf("%u, ", arr[i]);
	}
	printf("%u]\n", arr[len-1]);
}

// Print an array of ints in [,,] format
void printArray(int *arr, int len){
	printf("[");
	for (int i = 0; i < len -1; ++i) {
		printf("%d, ", arr[i]);
	}
	printf("%d]\n", arr[len-1]);
}

// A simple helper function to compute the difference between to points in time
double time_diff(struct timeval x , struct timeval y){
	double x_ms , y_ms , diff;
	 
	x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
	y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;
	 
	diff = (double)y_ms - (double)x_ms;
	 
	return diff;
}

// A simple helper function to reset two arrays with random values
void resetTestData(float *floatArr, int lenFloats, int *intArr, int lenInts){
	for (int i = 0; i < lenFloats; i++){
		floatArr[i] = (float)rand()/(float)(RAND_MAX/1.0);
	}
	for (int i = 0; i < lenInts; i++){
		intArr[i] = (int)rand()/(float)(RAND_MAX/10);
	}
}

int main(int argc, char ** argv) {

	// not testing ints so set it to length 0
	int intArr[0] = {};
	int lenInts = 0;
	int * d_ints;
	cudaMalloc((void **) &d_ints, lenInts);


	// set the size to modulus by
	int size = 500; 

	// number of trials for each configuration of tiles and floats
	int numTrials = 5000; 
	int maxTilings = 1025; // this is the limit of the number of tilings

	struct timeval beforeParallel, afterParallel, beforeSerial, afterSerial;

	// jump up powers of 2
	for (int numtilings = 16; numtilings < maxTilings; numtilings*=2){

		// init the tilings here as it wont change size for the inner loops
		unsigned tileIdx[numtilings];
		unsigned testTileIdx[numtilings];

		unsigned * d_hashArray;
		cudaMalloc((void **) &d_hashArray, numtilings*sizeof(unsigned));


		int incrementBy = (int) numtilings/16; // have 4 data points per numtiling

		for (int lenFloats = 1; lenFloats < (int) numtilings/4; lenFloats+=incrementBy){ // make sure there are more than 4x as many tiles
			float floatArr[lenFloats];

			float * d_floats;
			cudaMalloc((void **) &d_floats, lenFloats * sizeof(float));

			double sumParallel = 0.0, avgTimeParallel = 0.0, minTimeParallel = INFINITY, maxTimeParallel = 0.0, sumSerial = 0.0, avgTimeSerial = 0.0, minTimeSerial = INFINITY, maxTimeSerial = 0.0;

			for (int trial = 0; trial < numTrials; trial++){

				// reset float array
				resetTestData(floatArr, lenFloats, intArr, lenInts);

				// time the Parallel tiles
				gettimeofday(&beforeParallel , NULL);
				parallel_tiles(size, d_hashArray, numtilings, d_floats, floatArr, lenFloats, d_ints, intArr, lenInts, tileIdx);
				gettimeofday(&afterParallel , NULL);

				// time the Serial tiles
				gettimeofday(&beforeSerial, NULL);
				tiles(size, numtilings, floatArr, lenFloats, intArr, lenInts, testTileIdx);
				gettimeofday(&afterSerial, NULL);

				// confirm correct calculation
				int Errors = 0;
				for (int j = 0; j < numtilings; j++){
					if (tileIdx[j] != testTileIdx[j]){
						printf("Error: Incorrect Arrays\nCorrect Array:  ");
						printUnsignedArray(testTileIdx, numtilings);
						printf("\nComputed Array: ");
						printUnsignedArray(tileIdx, numtilings);
						Errors = 1;
						break;
					}
				}
				if (Errors){
					// if there is an error (differing arrays), free the memory and print debug info 
					cudaFree(d_floats);
					cudaFree(d_hashArray);
					cudaFree(d_ints);
					printf("Error: tilings %d, lenFloats %d, trial %d\n", numtilings, lenFloats, trial);
					return 1;
				}

				// compute time comparison
				double timeTakenParallel = time_diff(beforeParallel , afterParallel);
				sumParallel += timeTakenParallel;

				if (timeTakenParallel < minTimeParallel){
					minTimeParallel = timeTakenParallel;
				}
				if (timeTakenParallel > maxTimeParallel){
					maxTimeParallel = timeTakenParallel;
				}

				double timeTakenSerial = time_diff(beforeSerial , afterSerial);
				sumSerial += timeTakenSerial;

				if (timeTakenSerial < minTimeSerial){
					minTimeSerial = timeTakenSerial;
				}
				if (timeTakenSerial > maxTimeSerial){
					maxTimeSerial = timeTakenSerial;
				}

			} // trialsloop

			// compute the average time for each scenario
			avgTimeParallel = sumParallel/numTrials;
			avgTimeSerial = sumSerial/numTrials;

			// if the parallel time is less than the serial time print the details and break the loop
			// because increasing the number of floats will make serial take longer
			if (avgTimeParallel < avgTimeSerial){
							printf("tilings %d, lenFloats %d\n\tSERIAL\n\t\tAvg time : %.0lf us\tMin Time : %.0lf us\tMax Time : %.0lf us\n\tPARALLEL\n\t\tAvg time : %.0lf us\tMin Time : %.0lf us\tMax Time : %.0lf us\n\n", numtilings, lenFloats, avgTimeSerial, minTimeSerial, maxTimeSerial, avgTimeParallel, minTimeParallel, maxTimeParallel); 
				printf("---------------------------------------------------------\n");
				break;
			}

			// free the float array because it is about to be resized
			cudaFree(d_floats);

		} // lenFloatsloop

		// free the hashArray as it is about to be resized
		cudaFree(d_hashArray);

	} // numtilingsloop
	
	// free the int array
	cudaFree(d_ints);

	return 0;
}