import numpy as _np
import ot as _ot
import collections as _collections

def compute_optimal_transport_flux(
	distributions_to: _np.array, distributions_from: _np.array, distance_mat: _np.array,
	ot_solve_kwargs : dict={}, force_for_loop : bool=True
):
	'''
	The compute_distance_matrix function computes the distance between a list of coordinates.

	Parameters:
		distributions_to (np.array): 2d-array of shape (`num_categories`, `size`) or 1d-array of length `size`
			representing the end distribution of population.
		distributions_from (np.array): 2d-array of shape (`num_categories`, `size`) or 1d-array of length `size`
			representing the starting distribution of population that will be transported to distributions_to.

	Optional parameters:
		distance_mat (np.array): 2d-array of shape (`size`, `size`) filled with the distance between each location.
		ot_solve_kwargs (dict): list of additional amed argument to pass to the ot.solve function that is used as a backend.
		force_for_loop (bool): force solving using ot.solve instead of ot.solve_batch.

	Returns:
		transport_plane (np.array): either a 3d array of shape (`num_categories`, `size`, `size`) or a 2d array
			of shape (`size`, `size`) if distributions_from is only 1d. Element of index (n, i, j) reprensents
			the flux of population n from locality i to locality j.
	'''

	is_distributions_from_1d = not isinstance(distributions_from[0], _collections.abc.Iterable)
	is_distributions_to_1d   = not isinstance(distributions_to[0],   _collections.abc.Iterable)
	
	if is_distributions_from_1d:
		assert is_distributions_to_1d, "If the distribution (distributions_from) is 1-dimensional, then the null distribution (distributions_to) must also e one-dimensional"

	if is_distributions_from_1d == 1:
		size, num_categories = len(distributions_from), 1
	else:
		size, num_categories = len(distributions_from[0]), len(distributions_from)


	assert len(distributions_from.shape) <= 2, f"The distribution (distributions_from) can only be 1 or 2-dimensional, the shape given was { distributions_from.shape }"
	assert distributions_to.shape in [(size,), (num_categories, size)], f"The null distribution (distributions_to) must be of shape ({ size },) or ({ num_categories }, { size }) given the distribution was of shape { distributions_from.shape }, the shape given was { distributions_to.shape }"
	assert distance_mat.shape == (size, size), f"The distance matrix (distance_mat) must be a square matrix of shape ({ size }, { size }) given the distribution was of shape { distributions_from.shape }, the shape given was { distance_mat.shape }"

	transport_plane = _np.zeros((num_categories, size, size)) if not is_distributions_from_1d else _np.zeros((num_categories, size, size))

	if is_distributions_from_1d:
		transport_plane = _ot.solve(distance_mat, distributions_to if is_distributions_to_1d else distributions_to[0], distributions_from, **ot_solve_kwargs).plan
	else:
		if force_for_loop:
			for dimension in range(num_categories):
				transport_plane[dimension, :, :] = _ot.solve(distance_mat, distributions_to if is_distributions_to_1d else distributions_to[dimension], distributions_from[dimension], **ot_solve_kwargs).plan
		else:
			distance_mat_batch = _np.repeat(_np.expand_dims(distance_mat, axis=0), num_categories, axis=0)
			if is_distributions_to_1d:
				distributions_to_batch = _np.repeat(_np.expand_dims(distributions_to, axis=0), num_categories, axis=0)
			else:
				distributions_to_batch = distributions_to
			transport_plane = _ot.solve_batch(
				M=distance_mat_batch, a=distributions_to_batch, b=distributions_from,
				reg=1 if "reg" not in ot_solve_kwargs else ot_solve_kwargs["reg"], **ot_solve_kwargs
			).plan
	return transport_plane


def compute_distance_matrix(coordinates: _np.array, exponent: float=2):
	'''
	The compute_distance_matrix function computes the distance between a list of coordinates.

	Parameters:
		coordinates (np.array): 2d-array of shape (`num_dimensions`, `size`) representing the position of each location.

	Optional parameters:
		exponent (float): the exponent used in the norm (2 is the euclidien norm).

	Returns:
		distance_mat (np.array): 2d-array of shape (`size`, `size`) filled with the distance between each location.
	'''

	assert len(coordinates.shape) == 2, f"coordinates passed to compute_distance_matrix must be a 2-dimensional array, array of shape { coordinates.shape } was given"

	size, num_dimensions = coordinates.shape[1], coordinates.shape[0]

	distance_mat = _np.zeros((size, size))
	for dimension in range(num_dimensions):
		distance_mat += _np.pow(_np.repeat(_np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - _np.repeat(_np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0), exponent)
	distance_mat = _np.pow(distance_mat, 1/exponent)

	return distance_mat

def compute_distance_matrix_polar(
	latitudes: _np.array, longitudes: _np.array,
	radius: float=6378137, unit: str="deg"
):
	'''
	The compute_distance_matrix_polar function computes the distance between a list of coordinates from polar
	coordinates on a sphere. by default it can be used for typical coordinates on earth.

	Parameters:
		latitudes (np.array): 1d-array of length `size` with the latitudes of each point.
		longitudes (np.array): 1d-array of length `size` with the longitudes of each point.

	Optional parameters:
		radius (float): radius of the sphere (by default 6378137 which is the radius of the earth in meters).
		unit (str): a string to define the unit of the longitude and latituden, eather "rad", "deg" (default),
		"arcmin", or "arcsec".

	Returns:
		distance_mat (np.array): 2d-array of shape (`size`, `size`) filled with the distance between each location.
	'''

	assert len(latitudes.shape) == 1, f"latitudes passed to compute_distance_matrix_polar must be a 1-dimensional array, array of shape { latitudes.shape } was given"
	assert len(longitudes.shape) == 1, f"longitudes passed to compute_distance_matrix_polar must be a 1-dimensional array, array of shape { longitudes.shape } was given"
	assert longitudes.shape == latitudes.shape, f"longitudes and latitudes passed to compute_distance_matrix_polar must match in size, { latitudes.shape } and { longitudes.shape } was given"

	size = len(latitudes)
	
	conversion_factors = {
		"rad"    : 1,
		"deg"    : _np.pi/180,
		"arcmin" : _np.pi/180/60,
		"arcsec" : _np.pi/180/3600,
	}

	assert unit in list(conversion_factors.keys()), f"unit passed to compute_distance_matrix_polar must be one of { list(conversion_factors.keys()) }, \"{ unit }\" was given"
	
	conversion_factor = conversion_factors[unit]

	latitudes_left   = _np.repeat(_np.expand_dims(latitudes,  axis=1), size, axis=1)*conversion_factor
	latitudes_right  = _np.repeat(_np.expand_dims(latitudes,  axis=0), size, axis=0)*conversion_factor
	longitudes_left  = _np.repeat(_np.expand_dims(longitudes, axis=1), size, axis=1)*conversion_factor
	longitudes_right = _np.repeat(_np.expand_dims(longitudes, axis=0), size, axis=0)*conversion_factor

	distance_mat = _np.sqrt(
		( latitudes_left  - latitudes_right )**2 +
		((longitudes_left - longitudes_right)**2)*_np.cos(latitudes_left)*_np.cos(latitudes_right)
	) * radius

	return distance_mat

def compute_unitary_direction_matrix(
	coordinates: _np.array, distance_mat: _np.array=None,
	exponent: float=2
):
	'''
	The compute_unitary_direction_matrix function computes the matrix of unitary vectors used to computed
	direction in the main functions.

	Parameters:
		coordinates (np.array): 2d-array of shape (`num_dimensions`, `size`) representing the position of each location.

	Optional parameters:
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

	assert len(coordinates.shape) == 2, f"coordinates passed to compute_unitary_direction_matrix must be a 2-dimensional array, array of shape { coordinates.shape } was given"

	size, num_dimensions = coordinates.shape[1], coordinates.shape[0]
	unitary_direction_matrix = _np.zeros((num_dimensions, size, size))

	if distance_mat is not None:
		assert distance_mat.shape == (size, size), f"distance matrix (distance_mat) passed to compute_unitary_direction_matrix must be of shape ({ size }, { size }) matching the length of coordinates, { distance_mat.shape } was given"

	distance_mat_is_None = distance_mat is None
	if distance_mat_is_None:
		distance_mat = compute_distance_matrix(coordinates, exponent)

	distance_mat_is_zero = distance_mat == 0
	distance_mat[distance_mat_is_zero] = 1
		
	for dimension in range(num_dimensions):
		unitary_direction_matrix[dimension, :, :] = (_np.repeat(_np.expand_dims(coordinates[dimension, :], axis=1), size, axis=1) - _np.repeat(_np.expand_dims(coordinates[dimension, :], axis=0), size, axis=0)) / distance_mat
		for i in range(size):
			unitary_direction_matrix[dimension, i, i] = 0

	unitary_direction_matrix[:, distance_mat_is_zero] = 0
	distance_mat[distance_mat_is_zero] = 0

	if distance_mat_is_None:
		return unitary_direction_matrix, distance_mat
	return unitary_direction_matrix

def compute_unitary_direction_matrix_polar(
	latitudes: _np.array, longitudes: _np.array,
	distance_mat: _np.array=None, radius: float=6378137, unit: str="deg"
):
	'''
	The compute_unitary_direction_matrix_polar function computes the matrix of unitary vectors used to computed
	direction in the main functions, between a list of coordinates from polar coordinates on a sphere. by default
	it can be used for typical coordinates on earth.

	Parameters:
		latitudes (np.array): 1d-array of length `size` with the latitudes of each point.
		longitudes (np.array): 1d-array of length `size` with the longitudes of each point.

	Optional parameters:
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

	assert len(latitudes.shape) == 1, f"latitudes passed to compute_unitary_direction_matrix_polar must be a 1-dimensional array, array of shape { latitudes.shape } was given"
	assert len(longitudes.shape) == 1, f"longitudes passed to compute_unitary_direction_matrix_polar must be a 1-dimensional array, array of shape { longitudes.shape } was given"
	assert longitudes.shape == latitudes.shape, f"longitudes and latitudes passed to compute_unitary_direction_matrix_polar must match in size, { latitudes.shape } and { longitudes.shape } was given"

	size = len(latitudes)

	if distance_mat is not None:
		assert distance_mat.shape == (size, size), f"distance matrix (distance_mat) passed to compute_unitary_direction_matrix_polar must be of shape ({ size }, { size }) matching the length of latitudes and longitudes, { distance_mat.shape } was given"

	conversion_factors = {
		"rad"    : 1,
		"deg"    : _np.pi/180,
		"arcmin" : _np.pi/180/60,
		"arcsec" : _np.pi/180/3600,
	}

	assert unit in list(conversion_factors.keys()), f"unit passed to compute_unitary_direction_matrix_polar must be one of { list(conversion_factors.keys()) }, \"{ unit }\" was given"
	
	conversion_factor = conversion_factors[unit]

	latitudes_left   = _np.repeat(_np.expand_dims(latitudes,  axis=1), size, axis=1)*conversion_factor
	latitudes_right  = _np.repeat(_np.expand_dims(latitudes,  axis=0), size, axis=0)*conversion_factor
	longitudes_left  = _np.repeat(_np.expand_dims(longitudes, axis=1), size, axis=1)*conversion_factor
	longitudes_right = _np.repeat(_np.expand_dims(longitudes, axis=0), size, axis=0)*conversion_factor

	unitary_direction_matrix = _np.zeros((2, size, size))

	distance_mat_is_None = distance_mat is None
	if distance_mat_is_None:
		distance_mat = _np.sqrt(
			(latitudes_left   - latitudes_right )**2 +
			((longitudes_left - longitudes_right)**2)*_np.cos(latitudes_left)*_np.cos(latitudes_right)
		) * radius

	distance_mat_is_zero = distance_mat == 0
	distance_mat[distance_mat_is_zero] = 1

	unitary_direction_matrix[0, :] = (latitudes_left  - latitudes_right ) * radius / distance_mat
	unitary_direction_matrix[1, :] = (longitudes_left - longitudes_right) * _np.sqrt(_np.cos(latitudes_left)*_np.cos(latitudes_right)) * radius / distance_mat

	unitary_direction_matrix[:, distance_mat_is_zero] = 0
	distance_mat[distance_mat_is_zero] = 0

	if distance_mat_is_None:
		return unitary_direction_matrix, distance_mat
	return unitary_direction_matrix

def compute_neighbor_index_matrix(distance_mat: _np.array):
	'''
	The compute_neighbor_index_matrix function computes the distance matrix corresponding to the neighbor index,
	i.e. the distance between i and j is N if there are exactly N-1 points closer to i than j. Note that this
	distance isn't symmetric.

	Parameters:
		distance_mat (np.array): 2d-array of shape (`size`, `size`) filled with the distance between each location.
	
	Returns:
		neighbor_index_mat (np.array): 2d-array of shape (`size`, `size`) filled with the neighbor index : the
			value at index (i, j) is N-1 if there are exactly N points closer to i than j.
	'''

	assert len(distance_mat.shape) == 2, f"distance matrix passed to compute_neighbor_index_matrix must be a 2-dimensional array, array of shape { distance_mat.shape } was given"
	assert distance_mat.shape[0] == distance_mat.shape[1], f"distance matrix passed to compute_neighbor_index_matrix must be a square matrix, array of shape { distance_mat.shape } was given"

	size           = distance_mat.shape[0]
	sorted_indeces = _np.argsort(distance_mat, axis=1)

	neighbor_index = _np.zeros_like(distance_mat)
	index_mat      = _np.repeat(_np.expand_dims(_np.arange(size), axis=0), size, axis=0)

	neighbor_index[index_mat.T, sorted_indeces] = index_mat

	return neighbor_index

def compute_cumulative_neighbor_cost(distance_mat: _np.array, measure: _np.array, conserve_order: bool=True):
	'''
	The compute_cumulative_neighbor_cost function computes the cumulative sum of value over the neighborhood index.
	i.e. the distance between i and j is the sum of the values located in the measure array for each lcoation k
	located between i and j included. Note that this distance isn't symmetric.

	Parameters:
		distance_mat (np.array): 2d-array of shape (`size`, `size`) filled with the distance between each location.
		measure (np.array): either an array of length `size`, or a 2d-array of shape (`size`, `size`) that will be
			accumulated. If measure is a 2d array, then the values accumulated between i and j will be the indeces
			(i, k) for for each lcoation k located between i and j included.

	Optional parameters:
		conserve_order (bool): wether to keep the order or to sort the array, default is true
	
	Returns:
		cumulative_neighbor_cost (np.array): 2d-array of shape (`size`, `size`) filled with the accumulated values
			located in the array measure according to the neighborhood index.
	'''
	assert len(distance_mat.shape) == 2, f"distance matrix passed to compute_cumulative_neighbor_cost must be a 2-dimensional array, array of shape { distance_mat.shape } was given"
	assert distance_mat.shape[0] == distance_mat.shape[1], f"distance matrix passed to compute_cumulative_neighbor_cost must be a square matrix, array of shape { distance_mat.shape } was given"

	size = distance_mat.shape[0]

	assert len(measure.shape) <= 2, f"measure array passed to compute_cumulative_neighbor_cost must be a 1 or 2-dimensional array, array of shape { measure.shape } was given"

	sorted_indeces = _np.argsort(distance_mat, axis=1)

	accumulated_values = _np.zeros_like(distance_mat)
	index_mat          = _np.repeat(_np.expand_dims(_np.arange(size), axis=0), size, axis=0)

	if len(measure.shape) == 2:
		assert measure.shape == (size, size), f"2d measure array passed to compute_cumulative_neighbor_cost must be of shape ({ size }, { size }) matching the size of the distance matrix, { measure.shape } was given"
		unsorted_accumulated_values = _np.cumsum(_np.take_along_axis(measure, sorted_indeces, axis=1), axis=1)
		if conserve_order:
			accumulated_values[index_mat.T, sorted_indeces] = unsorted_accumulated_values
			return accumulated_values
		return unsorted_accumulated_values

	assert len(measure) == size, f"1d measure array passed to compute_cumulative_neighbor_cost must be of length { size } matching the size of the distance matrix, an array of length { len(measure) } was given"
	unsorted_accumulated_values = _np.cumsum(measure[sorted_indeces], axis=1)
	if conserve_order:
		accumulated_values[index_mat.T, sorted_indeces] = unsorted_accumulated_values
		return accumulated_values
	return unsorted_accumulated_values