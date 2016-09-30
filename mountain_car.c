#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Implementation of mountiancar:
// The first index is the postion, the second is the velocity

void init(float *observation){
	// set observation to a random starting position
	observation[0] = -.6 + ((float)rand()/(float)(RAND_MAX/1.0))*.2;
	observation[1] = 0.0;
}



void sample(float *observation, float action, int *R){
	if (action < -1 || action > 1){
		printf("Invalid action: %.2f\n", action);
	}

	*R = -1;

	observation[1] += 0.001*action - 0.0025*cos(3*observation[0]);
	if (observation[1] < -0.07){
		observation[1] = -0.07;
	} else if (observation[1] >= 0.07){
		observation[1] = 0.06999999;
	}

	observation[0] += observation[1];

	if (observation[0] >= .5){
		// set the observations to 1 to indicate the end of the trial
		// rather than returning an 'is done' bool
		observation[0] = 1;
		observation[1] = 1;
	} else if (observation[0] < -1.2){
		observation[0] = -1.2;
		observation[1] = 0.0;
	}
}



// int main(int argc, char **argv){

// 	float observation[2] = {0.0, 0.0};

// 	// test init
// 	init(observation);
// 	printf("Init Test: [%.5f, %.5f]\n", observation[0], observation[1]);

// 	// test sample
// 	int r = 0;
// 	sample(observation, 1.0, &r);
// 	printf("Sample Test\n obs: [%.5f, %.5f], reward: %d\n", observation[0], observation[1], r);

// 	// test end condition
// 	observation[0] = .5;
// 	observation[1] = .07;
// 	sample(observation, 1.0, &r);
// 	printf("Sample Test\n obs: [%.5f, %.5f], reward: %d\n", observation[0], observation[1], r);

// }