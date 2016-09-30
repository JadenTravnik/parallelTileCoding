
int lenInts, lenFloats, numPrototypes, numCoordinates;
int *d_ints;
float *d_floats;
float *d_prototypes;
float *d_activationRadii;
int *d_features;
		

void initialize(int _lenInts, int _lenFloats, int _numPrototypes, float *h_activationRadii){

	lenInts = _lenInts;
	lenFloats = _lenFloats;
	numPrototypes = _numPrototypes;
	numCoordinates = _lenInts + _lenFloats;

	cudaMalloc((void **) &d_ints, lenInts*sizeof(int));
	cudaMalloc((void **) &d_floats, lenFloats * sizeof(float));	

	// initialize random prototypes
	float h_prototypes[numPrototypes*numCoordinates];
	for (int i = 0; i < numPrototypes*numCoordinates; i++){
		h_prototypes[i] = (float)rand()/(float)(RAND_MAX/1.0);
	}

	cudaMalloc((void **) &d_prototypes, numPrototypes*numCoordinates*sizeof(float));
	cudaMemcpy(d_prototypes, h_prototypes, numPrototypes * numCoordinates * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_features, numPrototypes*sizeof(int));

	cudaMalloc((void **) &d_activationRadii, lenFloats * sizeof(float));
	cudaMemcpy(d_activationRadii, h_activationRadii, lenFloats * sizeof(float), cudaMemcpyHostToDevice);
}


// threadIdx.x = coord
// blockIdx.x = prototype
__global__ void calcFeatures(float *d_prototypes, float *d_floats, int lenFloats, int *d_ints, int lenInts, float *d_activationRadii, int *d_features){

	float val = 0.0;
	
	if (threadIdx.x < lenFloats){

		float distance = fabsf(d_floats[threadIdx.x] - d_prototypes[blockIdx.x * (lenFloats + lenInts) + threadIdx.x]);
		val = distance <= d_activationRadii[threadIdx.x] ? 1 - distance/d_activationRadii[threadIdx.x] : 0;
	} else {
		float distance = fabsf(((float) d_ints[threadIdx.x - lenFloats]) - d_prototypes[blockIdx.x * (lenFloats + lenInts) + threadIdx.x]);
		val = distance <= d_activationRadii[threadIdx.x] ? 1 - distance/d_activationRadii[threadIdx.x] : 0;
	}

	atomicAnd(&d_features[blockIdx.x], val > 0 ? 1 : 0);

}

// TODO finish this
void parallel_getFeaturesActivationRadii(float *h_floatArr, int *h_intArr, int *h_features){

	cudaMemset(d_features, 0xF, numPrototypes*sizeof(int)); 

	cudaMemcpy(d_floats, h_floatArr, lenFloats*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ints, h_intArr, lenInts * sizeof(int), cudaMemcpyHostToDevice);

	calcFeatures<<<numPrototypes, numCoordinates>>>(d_prototypes, d_floats, lenFloats, d_ints, lenInts, d_activationRadii, d_features);

	cudaMemcpy(h_features, d_features, numPrototypes * sizeof(float), cudaMemcpyDeviceToHost);
}
