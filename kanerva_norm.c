#include <stdio.h>
#include <math.h>
#include <stdlib.h>


int numPrototypes;
int lenFloats;
int lenInts;
float threshold;

float prototypes[];


void initialize(int _lenInts, int _lenFloats, int _numPrototypes){

	lenInts = _lenInts;
	lenFloats = _lenFloats;
	numPrototypes = _numPrototypes;
	numCoordinates = _lenInts + _lenFloats;


	// initialize random prototypes
	prototypes[numPrototypes*numCoordinates];
	for (int i = 0; i < numPrototypes*numCoordinates; i++){
		prototypes[i] = (float)rand()/(float)(RAND_MAX/1.0);
	}
}


// returns the tile indicies corresponding to the floats and ints
void getFeaturesNorm(float *floats, int *ints, int *features) {
	
	for (int i = 0; i < numPrototypes; i++) {
		float *prototype = prototypes[i];
		float distance = 0.0;

		float diff = 0.0;

		// Compute using norm

		for (int j = 0; j < lenFloats; j++) {
			diff = floats[j] - prototype[j];
			distance += diff*diff;
		}

		for (int j = 0; j < lenInts; j++){
			diff = (float) ints[j] - prototype[j+lenFloats];
			distance += diff*diff;
		}

		if (sqrt(distance) < threshold){
			features[i] = 1;
		}
	}
}