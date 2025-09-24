import numpy as np

def compute_unitary_direction_matrix(coordinates, distance_mat=None):
	size, num_dimensions = len(coordinates[0]), len(coordinates)
	unitary_direction_matrix = np.zeros((num_dimensions, size, size))

	if distance_mat is None:
		distance_mat = np.zeros((size, size))
		for dimension in range(num_dimensions):
			distance_mat += np.pow(np.repeat(np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - np.repeat(np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0), 2)
		distance_mat = np.pow(distance_mat, 0.5)

	for dimension in range(num_dimensions):
		unitary_direction_matrix[dimension, :, :] = (np.repeat(np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - np.repeat(np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0)) / distance_mat
		for i in range(size):
			unitary_direction_matrix[dimension, i, i] = 0

	return unitary_direction_matrix