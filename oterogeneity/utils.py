import numpy as np

def compute_distance_matrix(coordinates, exponent: float=2):
	'''
    The compute_distance_matrix function computes the distance between a list of coordinates.

    Parameters:
        coordinates (np.array): 2d-array of shape (`num_dimensions`, `size`) representing the position of each location.
        exponent (float): the exponent used in the norm (2 is the euclidien norm).

	Returns:
		distance_mat (np.array): 2d-array of shape (`size`, `size`) filled with the distance between each location.
    '''

	size, num_dimensions = len(coordinates[0]), len(coordinates)

	distance_mat = np.zeros((size, size))
	for dimension in range(num_dimensions):
		distance_mat += np.pow(np.repeat(np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - np.repeat(np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0), exponent)
	distance_mat = np.pow(distance_mat, 1/exponent)

	return distance_mat

def compute_distance_matrix_polar(latitudes, longitudes, radius: float=6378137, unit: str="deg"):
	'''
	The compute_distance_matrix_polar function computes the distance between a list of coordinates from polar
	coordinates on a sphere. by default it can be used for typical coordinates on earth.

	Parameters:
        latitudes (np.array): 1d-array of length `size` with the latitudes of each point.
        longitudes (np.array): 1d-array of length `size` with the longitudes of each point.
        radius (float): radius of the sphere (by default 6378137 which is the radius of the earth in meters).
        unit (str): a string to define the unit of the longitude and latituden, eather "rad", "deg" (default),
        "arcmin", or "arcsec".

	Returns:
		distance_mat (np.array): 2d-array of shape (`size`, `size`) filled with the distance between each location.
	'''

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

def compute_unitary_direction_matrix(coordinates, distance_mat=None, exponent: float=2):
	'''
	The compute_unitary_direction_matrix function computes the matrix of unitary vectors used to computed
	direction in the main functions.

	Parameters:
        coordinates (np.array): 2d-array of shape (`num_dimensions`, `size`) representing the position of each location.
        distance_mat (np.array): you can optionally pass a 2d-array of shape (`size`, `size`) filled with the distance
        	between each location. If not passed it will be computed and returned.
        exponent (float): the exponent used in the norm (2 is the euclidien norm). If a distance matrix is passed, it
        	must have been computed with the same exponent as the one passed to this function.
	
	Returns:
		unitary_direction_matrix (np.array): 3d-array of shape (`num_categories`, `size`, `size`) representing the
			unitary vector between each location.
		distance_mat (np.array): a distance matrix is returned if it was not passed as a parameter (to avoid
			recomputing it), it is a 2d-array of shape (`size`, `size`) filled with the distance between
			each location.
	'''

	size, num_dimensions = len(coordinates[0]), len(coordinates)
	unitary_direction_matrix = np.zeros((num_dimensions, size, size))

	distance_mat_is_None = distance_mat is None
	if distance_mat_is_None:
		distance_mat = compute_distance_matrix(coordinates, exponent)

	distance_mat_is_zero = distance_mat == 0
	distance_mat[distance_mat_is_zero] = 1
		
	for dimension in range(num_dimensions):
		unitary_direction_matrix[dimension, :, :] = (np.repeat(np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - np.repeat(np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0)) / distance_mat
		for i in range(size):
			unitary_direction_matrix[dimension, i, i] = 0

	unitary_direction_matrix[:, distance_mat_is_zero] = 0
	distance_mat[distance_mat_is_zero] = 0

	if distance_mat_is_None:
		return unitary_direction_matrix, distance_mat
	return unitary_direction_matrix

def compute_unitary_direction_matrix_polar(latitudes, longitudes, distance_mat=None, radius: float=6378137, unit: str="deg"):
	'''
	The compute_unitary_direction_matrix_polar function computes the matrix of unitary vectors used to computed
	direction in the main functions, between a list of coordinates from polar coordinates on a sphere. by default
	it can be used for typical coordinates on earth.

	Parameters:
        latitudes (np.array): 1d-array of length `size` with the latitudes of each point.
        longitudes (np.array): 1d-array of length `size` with the longitudes of each point.
        distance_mat (np.array): you can optionally pass a 2d-array of shape (`size`, `size`) filled with the distance
        	between each location. If not passed it will be computed and returned.
        radius (float): radius of the sphere (by default 6378137 which is the radius of the earth in meters).
        unit (str): a string to define the unit of the longitude and latituden, eather "rad", "deg" (default),
        "arcmin", or "arcsec".
	
	Returns:
		unitary_direction_matrix (np.array): 3d-array of shape (`num_categories`, `size`, `size`) representing the
			unitary vector between each location.
		distance_mat (np.array): a distance matrix is returned if it was not passed as a parameter (to avoid
			recomputing it), it is a 2d-array of shape (`size`, `size`) filled with the distance between
			each location.
	'''

	size = len(latitudes)

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

	unitary_direction_matrix = np.zeros((2, size, size))

	distance_mat_is_None = distance_mat is None
	if distance_mat_is_None:
		distance_mat = np.sqrt(
			(latitudes_left   - latitudes_right )**2 +
			((longitudes_left - longitudes_right)**2)*np.sin(latitudes_left)*np.sin(latitudes_right)
		) * radius

	distance_mat_is_zero = distance_mat == 0
	distance_mat[distance_mat_is_zero] = 1

	unitary_direction_matrix[0, :] = (latitudes_left  - latitudes_right ) * radius / distance_mat
	unitary_direction_matrix[1, :] = (longitudes_left - longitudes_right) * np.sqrt(np.sin(latitudes_left)*np.sin(latitudes_right)) * radius / distance_mat

	unitary_direction_matrix[:, distance_mat_is_zero] = 0
	distance_mat[distance_mat_is_zero] = 0

	if distance_mat_is_None:
		return unitary_direction_matrix, distance_mat
	return unitary_direction_matrix