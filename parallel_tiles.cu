/*
Parallel Tile Coding Software version 3.0beta translated to C
by Jaden Travnik based on Rich Sutton's Python implementation
*/


// PARALLEL TILECODING
// A gpu kernal function which completes the hashing function started in calCoordAndHashFloat and stores it in d_hashArray on the gpu
__global__ void shiftHash(unsigned * d_hashArray, int size){
	unsigned h = d_hashArray[threadIdx.x];

	h ^= ((unsigned) h >> 20) ^ ((unsigned)h >> 12);
	h = h ^ ((unsigned)h >> 7) ^ ((unsigned)h >> 4);

	d_hashArray[threadIdx.x] = h % size;
}

// A gpu kernel function which takes the floats, and integers and finds hashes their value 
// blockIdx.x is the tile index, threadIdx.x is the index of the coordinate
// d_hashArray is an array of length numTiles, it will contain the hashed index
__global__ void calcCoordAndHashFloat(unsigned * d_hashArray, int numtilings, float * d_floats, int lenFloats, int * d_ints, int lenInts){

	float coord = 1;
	unsigned floatAsInt = 1;
	int offset = threadIdx.x - 2; // these offsets are used to get the same hash results from the serial hashFloatArray function

	if (threadIdx.x > 0){
		coord = (float) blockIdx.x;

		if (threadIdx.x > 1 && threadIdx.x <= lenFloats + 1){

			 // this is the tile mapping function of floating point numbers
			coord = floor((floor(d_floats[offset]*numtilings) + blockIdx.x + offset*blockIdx.x*2) / numtilings);

		} else if (threadIdx.x > lenFloats + 1) {

			// "append" the integers to the coordinate array
			coord = (float) d_ints[offset - lenFloats]; 

		}

		// This is where the hashing function starts. Another hashing function could be used instead and here is where it would be used.
		floatAsInt = *(int*)(&coord); 
	}

	for (int i = 0; i < lenFloats + lenInts + 1 - threadIdx.x; i++){
		floatAsInt *= 31; // cant use pow on the gpu as we want an unsigned overflow
	}

	atomicAdd(& d_hashArray[blockIdx.x], floatAsInt);
}

// returns tile indicies corresponding to the floats and ints
void parallel_tiles(int size, unsigned * d_hashArray, int numtilings, float * d_floats, float *h_floats, int lenFloats, int * d_ints, int *h_ints, int lenInts, unsigned *Tiles) {

	// reset the d_hashArray to be zero so we can add to it in calcCoordAndHashFloat
	cudaMemset(d_hashArray, 0, numtilings*sizeof(unsigned)); 

	// Copy the data from the cpu over to the gpu
	cudaMemcpy(d_floats, h_floats, lenFloats * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_ints, h_ints, lenInts * sizeof(int), cudaMemcpyHostToDevice);
	
	// call kernel to compute the coordinates and being the hashing function
	calcCoordAndHashFloat<<<numtilings, 2 + lenFloats + lenInts>>>(d_hashArray, numtilings, d_floats, lenFloats, d_ints, lenInts);
	
	// call a kernel to complete the hashing function
	shiftHash<<<1,numtilings>>>(d_hashArray, size);

	// copy the memory from the gpu to the cpu
	cudaMemcpy(Tiles, d_hashArray, numtilings * sizeof(int), cudaMemcpyDeviceToHost);
}