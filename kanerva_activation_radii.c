// returns the tile indicies corresponding to the floats and ints
void getFeaturesActivationRadii(int numPrototypes, int numCoordinates, float *prototypes,float *floats, int lenFloats, int *ints, int lenInts, float *activationRadii, int *features) {


	for (int i = 0; i < numPrototypes; i++) {

		float minValue = INFINITY;
		float distance;
		float val;

		// Do floats
		for (int j = 0; j < lenFloats; j++) {

			distance = fabs(floats[j] - prototypes[i*lenFloats + j]);

			val = distance <= activationRadii[j] ? 1 - distance/activationRadii[j] : 0;


			minValue = minValue < val ? minValue : val;
		}

		// Do ints
		for (int j = 0; j < lenInts; j++) {
			distance = fabs((float)ints[j] - prototypes[i*lenFloats + j]);

			val = distance <= activationRadii[j + lenFloats] ? 1 - distance/activationRadii[j + lenFloats] : 0;


			minValue = minValue < val ? minValue : val;
		}

		// if close enough, activate feature
		features[i] = minValue > 0 ? 1 : 0;
	}
}
