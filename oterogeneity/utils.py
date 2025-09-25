import numpy as np

def compute_distance_matrix(coordinates, exponent=2):
	size, num_dimensions = len(coordinates[0]), len(coordinates)

	distance_mat = np.zeros((size, size))
	for dimension in range(num_dimensions):
		distance_mat += np.pow(np.repeat(np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - np.repeat(np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0), exponent)
	distance_mat = np.pow(distance_mat, 1/exponent)

	return distance_mat

def compute_distance_matrix_polar(latitudes, longitudes, radius=6378137, unit="deg"):
	conversion_factor = {
		"rad"    : 1,
		"deg"    : np.pi/180,
		"arcmin" : np.pi/180/60,
		"arcsec" : np.pi/180/3600,
	}[unit]

	latitudes_left   = np.repeat(np.expand_dims(latitudes,  axis=1), size, axis=1)*conversion_factor
	latitudes_right  = np.repeat(np.expand_dims(latitudes,  axis=0), size, axis=0)*conversion_factor
	longitudes_left  = np.repeat(np.expand_dims(longitudes, axis=1), size, axis=1)*conversion_factor
	longitudes_right = np.repeat(np.expand_dims(longitudes, axis=0), size, axis=0)*conversion_factor

	distance_mat = np.sqrt(
		(latitudes_left - latitudes_right)**2 +
		((latitudes_left - latitudes_right)**2)*longitudes_left*longitudes_right
	) * radius

	return distance_mat

def compute_unitary_direction_matrix(coordinates, distance_mat=None, exponent=2):
	size, num_dimensions = len(coordinates[0]), len(coordinates)
	unitary_direction_matrix = np.zeros((num_dimensions, size, size))

	distance_mat_is_None = distance_mat is None
	if distance_mat_is_None:
		distance_mat = compute_distance_matrix(coordinates, exponent)
		
	for dimension in range(num_dimensions):
		unitary_direction_matrix[dimension, :, :] = (np.repeat(np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - np.repeat(np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0)) / distance_mat
		for i in range(size):
			unitary_direction_matrix[dimension, i, i] = 0

	if distance_mat_is_None:
		return unitary_direction_matrix, distance_mat
	return unitary_direction_matrix

def compute_unitary_direction_matrix_polar(latitudes, longitudes, distance_mat=None, radius=6378137, unit="deg"):
	size, num_dimensions = len(coordinates[0]), len(coordinates)

	conversion_factor = {
		"rad"    : 1,
		"deg"    : np.pi/180,
		"arcmin" : np.pi/180/60,
		"arcsec" : np.pi/180/3600,
	}[unit]

	latitudes_left   = np.repeat(np.expand_dims(latitudes,  axis=1), size, axis=1)*conversion_factor
	latitudes_right  = np.repeat(np.expand_dims(latitudes,  axis=0), size, axis=0)*conversion_factor
	longitudes_left  = np.repeat(np.expand_dims(longitudes, axis=1), size, axis=1)*conversion_factor
	longitudes_right = np.repeat(np.expand_dims(longitudes, axis=0), size, axis=0)*conversion_factor

	unitary_direction_matrix = np.zeros((num_dimensions, size, size))

	distance_mat_is_None = distance_mat is None
	if distance_mat_is_None:
		distance_mat = np.sqrt(
			(latitudes_left - latitudes_right)**2 +
			((latitudes_left - latitudes_right)**2)*np.sin(longitudes_left)*np.sin(longitudes_right)
		) * radius

	unitary_direction_matrix[0, :] = (latitudes_left - latitudes_right) * radius / distance_mat
	unitary_direction_matrix[1, :] = (latitudes_left - latitudes_right) * np.sqrt(np.sin(longitudes_left)*np.sin(longitudes_right)) * radius / distance_mat

	if distance_mat_is_None:
		return unitary_direction_matrix, distance_mat
	return unitary_direction_matrix