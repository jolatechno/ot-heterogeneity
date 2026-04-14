import numpy as _np

def KL_divergence(array_A: _np.array, array_B: _np.array, min_value_avoid_zeros: float=1e-10):
	'''
	The KL_divergence function computes the KL divergence between two array (either 1d or 2d). If one or both array
	are 2 dimensionnal, then the return is a 1d vector of the divergence between each vector entry, otherwise if
	both array are 1 dimensional then the return is a float.

	Parameters:
        array_A (np.array): 2d-array of shape (`size`, `num_dimension`) or 1d vector of legth `num_dimension`
        array_B (np.array): 2d-array of shape (`size`, `num_dimension`) or 1d vector of legth `num_dimension`

    Optional parameters:
    	min_value_avoid_zeros (float): minimum value applied to array to avoid division by zero, default is 1e-10
	
	Returns:
		divergence (np.array): array containing the divergence between entry, either a 1d array of length
			`size` if one of the array is 2 dimensional, or a single float otherwise.
	'''

	assert len(array_A.shape) <= 2 and len(array_B.shape) <= 2, f"array passed to KL_divergence must be 1 or 2 dimensional, arrays of shape { array_A.shape } and { array_B.shape } were passed"

	is_array_A_1d = len(array_A.shape) == 1
	is_array_B_1d = len(array_B.shape) == 1
	is_output_1d  =  is_array_A_1d and is_array_B_1d
	num_dimension = len(array_A) if is_array_A_1d else array_A.shape[1]

	if is_output_1d:
		assert len(array_A) == len(array_B), f"1d array passed to KL_divergence must must be of the same shape, arrays of shape { array_A.shape } and { array_B.shape } were passed"

		return _np.sum(array_A * _np.log2(_np.maximum(array_A, min_value_avoid_zeros) / _np.maximum(array_B, min_value_avoid_zeros)))
	else:
		size     = array_A.shape[0] if is_array_A_1d else array_B.shape[0]
		array_A_ = array_A if not is_array_A_1d else _np.repeat(_np.expand_dims(array_A, axis=0), size, axis=0)
		array_B_ = array_B if not is_array_B_1d else _np.repeat(_np.expand_dims(array_B, axis=0), size, axis=0)

		assert array_A_.shape == array_B_.shape, f"array passed to KL_divergence must must be of the compatible shape, arrays of shape { array_A.shape } and { array_B.shape } were passed"

		return _np.sum(array_A * _np.log2(_np.maximum(array_A, min_value_avoid_zeros) / _np.maximum(array_B, min_value_avoid_zeros)), axis=1)

def total_variation(array_A : _np.array, array_B : _np.array):
	'''
	The KL_divergence function computes the KL divergence between two array (either 1d or 2d). If one or both array
	are 2 dimensionnal, then the return is a 1d vector of the divergence between each vector entry, otherwise if
	both array are 1 dimensional then the return is a float.

	Parameters:
        array_A (np.array): 2d-array of shape (`size`, `num_dimension`) or 1d vector of legth `num_dimension`
        array_B (np.array): 2d-array of shape (`size`, `num_dimension`) or 1d vector of legth `num_dimension`
	
	Returns:
		divergence (np.array): array containing the divergence between entry, either a 1d array of length
			`size` if one of the array is 2 dimensional, or a single float otherwise.
	'''

	assert len(array_A.shape) <= 2 and len(array_B.shape) <= 2, f"array passed to total_variation must be 1 or 2 dimensional, arrays of shape { array_A.shape } and { array_B.shape } were passed"
	
	is_array_A_1d = len(array_A.shape) == 1
	is_array_B_1d = len(array_B.shape) == 1
	is_output_1d  =  is_array_A_1d and is_array_B_1d
	num_dimension = len(array_A) if is_array_A_1d else array_A.shape[1]

	if is_output_1d:
		assert len(array_A) == len(array_B), f"1d array passed to total_variation must must be of the same shape, arrays of shape { array_A.shape } and { array_B.shape } were passed"

		return _np.sum(_np.abs(array_A - array_B, minimum))
	else:
		size     = array_A.shape[0] if is_array_A_1d else array_B.shape[0]
		array_A_ = array_A if not is_array_A_1d else _np.repeat(_np.expand_dims(array_A, axis=0), size, axis=0)
		array_B_ = array_B if not is_array_B_1d else _np.repeat(_np.expand_dims(array_B, axis=0), size, axis=0)
		
		assert array_A_.shape == array_B_.shape, f"array passed to total_variation must must be of the compatible shape, arrays of shape { array_A.shape } and { array_B.shape } were passed"

		return _np.sum(_np.abs(array_A_ - array_B_, minimum), axis=1)