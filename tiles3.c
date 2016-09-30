/*
Serial Tile Coding Software version 3.0beta translated to C
by Jaden Travnik based on Rich Sutton's Python implementation
*/


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// SERIAL TILECODING
// A hash function for an array of floats
unsigned hashFloatArray(float *arr, int len) {
    unsigned h = 1;

    for (int i = 0; i < len; i++){
    	int floatAsInt = *(int*)(&arr[i]);
        h = 31 * h + floatAsInt;
    }

    h ^= ((unsigned) h >> 20) ^ ((unsigned)h >> 12);
    h = h ^ ((unsigned)h >> 7) ^ ((unsigned)h >> 4);
    return h;
}

// a wrapper for the hash function that uses modulus on the size
unsigned hashcoords(float *coordinates, int numCoordinates, int size){
	int hashValue = 0;
	if (size > 0){
		hashValue = hashFloatArray(coordinates, numCoordinates) % size;
	}
	return hashValue;
}

// returns the tile indicies corresponding to the floats and ints
void tiles(int size, int numtilings, float *floats, int lenFloats, int *ints, int lenInts, unsigned *Tiles) {
	
	int numCoordinates = 1 + lenFloats + lenInts;
	float coords[numCoordinates];

	for (int i = 0; i < lenInts; i++){
		coords[i + 1 + lenFloats] = (float) ints[i];
	}

	float qfloats[lenFloats];
	for (int i = 0; i < lenFloats; ++i) {
		qfloats[i] = floor(floats[i]*numtilings);
	}

	// Python: for tiling in range(numtilings):
	for (int i = 0; i < numtilings; ++i) {

		int tilingX2 = i * 2;
		coords[0] = i;

		// Python: for q in qfloats:
		for (int j = 0; j < lenFloats; ++j) {
			float coord = floor((qfloats[j] + i + j*tilingX2) / numtilings);
			coords[j + 1] = coord;
		}

		Tiles[i] = hashcoords(coords, numCoordinates, size);
	}
}

// returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats
void tileswrap(int size, int numtilings, float *floats, int lenFloats, int *wrapwidths, int *ints, int lenInts, int *Tiles) {
	
	int numCoordinates = 1 + lenFloats + lenInts;

	// Python: qfloats = [floor(f*numtilings) for f in floats]
	float qfloats[lenFloats];
	for (int i = 0; i < lenFloats; ++i) {
		qfloats[i] = floor(floats[i]*numtilings);
	}

	// Python: for tiling in range(numtilings):
	for (int i = 0; i < numtilings; ++i) {
		int tilingX2 = i * 2;

		float coords[numCoordinates];
		coords[0] = i;
		int b = i;

		// Python: for q, width in zip_longest(qfloats, wrapwidths):
		for (int j = 0; j < lenFloats; ++j) {
			

			// Python: c = (q + b%numtilings) // numtilings
			int c = (int)floor((qfloats[j] + b%numtilings) / numtilings);
			
            // Python: coords.append(c%width if width else c)

			if (wrapwidths[j] > 0){
				coords[j + 1] = c % wrapwidths[j];
			} else {
				coords[j + 1] = c;
			}

			b += tilingX2;
		}

		// Python: coords.extend(ints)
		if (lenInts > 0){
			for (int j = 0; j < lenInts; ++j) {
				coords[j + 1 + lenFloats] = (float) ints[j];
			}
		}

		// Debugging: printFloatArray(coords, numCoordinates);

		Tiles[i] = hashcoords(coords, numCoordinates, size);
	}
}
