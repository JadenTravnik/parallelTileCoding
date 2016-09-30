#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include "tiles3.c"
#include "parallelTiles.cu"
#include "kanerva.c"



float multiplyVectors(float *a, float *b, int len){
	float c = 0;
	for (int i = 0; i < len; i++){
		c = a[i]*b[i];
	}
	return c;
}

void updateTraces(float *e, float *features, int numFeatures, float alpha, float lambda, float gamma){

	// (1 - alpha*gamma*lambda*e_transpose*features)
	float term = 1 - alpha*gamma*lambda*multiplyVectors(e, features);
	for (int i = 0; i < numFeatures; i++){
		e[i] = gamma*lambda*e[i] + term*features[i];
	}
}


void updateWeights(float *theta, float *e, int numFeatures, float alpha, float delta, float *features, float V, float V_old){

	// alpha*(delta + V - V_old)
	float  term1 = alpha*(delta + V - V_old);

	// alpha*(V - V_old)
	float term2 = alpha*(V - V_old);

	for (int i = 0; i < numFeatures; i++){
		theta[i] = theta[i] + term1*e[i] - term2*features[i];
	}

}

void learnUpdate(int numFeatures, float *features, float *oldFeatures, float *theta, float *e, float alpha, float lambda, float reward, float V_old){

	float V = multiplyVectors(theta, features, numFeatures);
	float V_prime = multiplyVectors(theta, oldFeatures);
	updateTraces(e, oldFeatures, numFeatures, alpha, lambda, gamma);
	float delta = reward + gamma*V_prime - V;
	updateWeights(theta, e, numFeatures, alpha, delta, oldFeatures, V, V_old);
	V_old = V;
	oldFeatures = features;
}