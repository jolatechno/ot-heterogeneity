import ot
import numpy as np
import sklearn
import collections
from . import utils

class ot_heterogeneity_results:
	'''
    The ot_heterogeneity_results class contains all of the results of a computation of spatial heterogeneity based on optimal
    transport using our method.

    Attributes:
        size (int): Number of spatial units (town, polling stations, etc...)
        num_categories (int): number of distinct categories
        num_dimensions (int): number of spacial dimensions (typically 2)
        has_direction (bool): whether the result contains directionality fields or not
        global_heterogeneity (float): global heterogeneity index
        global_heterogeneity_per_category (np.array): 1d-array of length `num_categories` that contains the local heterogeneity
        	index for each category.
        local_heterogeneity (np.array): 1d-array of length `size` that contains the local heterogeneity index for each location
        local_signed_heterogeneity (np.array): either a 2d-array of shape (`num_categories`, `size`) when `num_categories` > 1,
        	or a 1d-array of length `size` if `num_categories` = 1, that contains the signed heterogeneity index for each
        	category and each location.
		local_exiting_heterogeneity (np.array): 1d-array of length `size` that contains the heterogeneity index based only on
			exiting flux for each location.
		local_entering_heterogeneity (np.array): 1d-array of length `size` that contains the heterogeneity index based only on
			entering flux for each location.
		local_heterogeneity_per_category (np.array): 1d-array of length `size` that contains the heterogeneity index for each location.
		local_exiting_heterogeneity_per_category (np.array): 2d-array of shape (`num_categories`, `size`) that contains the
			heterogeneity index based only on exiting flux for each category and each location.
		local_entering_heterogeneity_per_category (np.array): 2d-array of shape (`num_categories`, `size`) that contains the
			heterogeneity index based only on entering flux for each category and each location.
		direction (np.array): 2d-array of shape (`num_dimensions`, `size`) representing the vectorial field of directionality.
		direction_per_category (np.array): 3d-array of shape (`num_categories`, `num_dimensions`, `size`) representing the
			vectorial field of directionality for each category.
    '''

	def __init__(self, size: int=0, num_categories: int=0, num_dimensions: int=1, has_direction : bool=False):
		if size <= 0:
			self.size, self.num_categories, self.num_dimensions, self.has_direction = size, 0, num_dimensions, False
			self.global_heterogeneity                      = None
			self.global_heterogeneity_per_category         = None
			self.local_heterogeneity                       = None
			self.local_signed_heterogeneity                = None
			self.local_exiting_heterogeneity               = None
			self.local_entering_heterogeneity              = None
			self.local_heterogeneity_per_category          = None
			self.local_exiting_heterogeneity_per_category  = None
			self.local_entering_heterogeneity_per_category = None
			self.direction                                 = None
			self.direction_per_category                    = None
		else:
			self.size, self.num_categories, self.num_dimensions, self.has_direction = size, num_categories, num_dimensions, has_direction
			self.global_heterogeneity         = 0
			self.local_heterogeneity          = np.zeros(size)
			self.local_exiting_heterogeneity  = np.zeros(size)
			self.local_entering_heterogeneity = np.zeros(size)

			if has_direction:
				self.direction = np.zeros((num_dimensions, size))
			else:
				self.num_dimensions         = 1
				self.direction              = None
				self.direction_per_category = None

			if num_categories <= 1:
				self.num_categories                            = 1
				self.local_signed_heterogeneity                = np.zeros(size)
				self.global_heterogeneity_per_category         = None
				self.local_heterogeneity_per_category          = None
				self.local_exiting_heterogeneity_per_category  = None
				self.local_entering_heterogeneity_per_category = None
				self.direction_per_category                    = None
			else:
				self.global_heterogeneity_per_category         = np.zeros( num_categories)
				self.local_signed_heterogeneity                = np.zeros((num_categories, size))
				self.local_heterogeneity_per_category          = np.zeros((num_categories, size))
				self.local_exiting_heterogeneity_per_category  = np.zeros((num_categories, size))
				self.local_entering_heterogeneity_per_category = np.zeros((num_categories, size))
				if has_direction:
					self.direction_per_category = np.zeros((num_categories, num_dimensions, size))


def ot_heterogeneity_from_null_distrib(
	distrib: np.array, null_distrib: np.array, distance_mat: np.array,
	transport_plane: np.array=None, return_transport_plane: bool=False,
	unitary_direction_matrix: np.array=None, local_weight_distrib: np.array=None, category_weights: np.array=None,
	epsilon_exponent: float=-1e-3, use_same_exponent_weight: bool=True,
	min_value_avoid_zeros: float=1e-5, ot_emb_args : list=[], ot_emb_kwargs : dict={}
):
	'''
    The ot_heterogeneity_from_null_distrib function is the most general function implementing our method for measuring
    spatial heterogeneity

    Parameters:
        distrib (np.array): 2d-array of shape (`num_categories`, `size`) representing the population distribution, i.e. the
        	population of each category in each location. 
        null_distrib (np.array): either a 2d-array of shape (`num_categories`, `size`) or a 1d-array of length `size` if
        	every category has the same null distribution, representing the null distribution (distribution without
        	heterogeneity), to which the distribution will be compared.
        distance_mat (np.array): 2d-array of shape (`size`, `size`) representing the distance between each locality.
        transport_plane (np.array): either a 3d array of shape (`num_dimensions`, `size`, `size`) or a 2d array
			of shape (`size`, `size`) if distributions_from is only 1d. Element of index (n, i, j) reprensents
			the flux of population n from locality i to locality j.
		return_transport_plane (bool): if true, the function will also return the transport plane.
        unitary_direction_matrix (np.array): 3d-array of shape (`num_dimensions`, `size`, `size`) representing the unitary
        	vector between each location.
        local_weight_distrib (np.array): 1d-array of length `size` representing the weight for each location. By default
        	this weight is simply the proportion of the total population located in each location.
        category_weights (np.array): 1d-array of length `num_categories` representing the weight for each num_category. By
        	default this weight is simply the proportion of the total population that belong to each category.
        epsilon_exponent (float): the distance matrix is exponentiated (element-wise) by an exponent 1+`epsilon_exponent`
        use_same_exponent_weight (bool): if true the cost (i.e. distant) is exponentiated by the same exponent as the one
        	for the cost matrix in the optimal-transport computation.
        min_value_avoid_zeros (float): value below wich a value is concidered zero.
        ot_emb_args (list): list of additional unamed argument to pass to the ot.emb function that is used as a backend.
        ot_emb_kwargs (dict): list of additional amed argument to pass to the ot.emb function that is used as a backend.

	Returns:
		results (ot_heterogeneity_results)
		transport_plane (np.array): either a 3d array of shape (`num_dimensions`, `size`, `size`) or a 2d array
			of shape (`size`, `size`) if distrib is only 1d. Element of index (n, i, j) reprensents the flux
			of population n from locality i to locality j. Returned only if return_transport_plane is true.
    '''

	is_local_weights_1dimensional = len(local_weight_distrib.shape) == 1 if local_weight_distrib is not None else False
	is_null_distrib_1dimensional  = len(null_distrib.shape)         == 1
	is_distrib_1dimensional       = len(distrib.shape)              == 1

	if is_distrib_1dimensional:
		assert is_null_distrib_1dimensional, "If the distribution is 1-dimensional, then the null distribution must also e one-dimensional"

	num_categories = 1 if is_distrib_1dimensional else distrib.shape[0]
	size           = distrib.shape[0] if is_distrib_1dimensional else distrib.shape[1]
	has_direction  = unitary_direction_matrix is not None
	num_dimensions = 1 if not has_direction else unitary_direction_matrix.shape[0]
	results        = ot_heterogeneity_results(size, num_categories, num_dimensions, has_direction)

	assert len(distrib.shape) <= 2, f"The distribution (distrib) can only be 1 or 2-dimensional, the shape given was { distrib.shape }"
	assert null_distrib.shape in [(size,), (num_categories, size)], f"The null distribution (null_distrib) must be of shape ({ size },) or ({ num_categories }, { size }) given the distribution was of shape { distrib.shape }, the shape given was { null_distrib.shape }"
	assert distance_mat.shape == (size, size), f"The distance matrix (distance_mat) must be a square matrix of shape ({ size }, { size }) given the distribution was of shape { distrib.shape }, the shape given was { distance_mat.shape }"
	if transport_plane is not None:
		if is_distrib_1dimensional:
			assert transport_plane.shape in [(size, size), (1, size, size)], f"The transport plane (transport_plane) passed must be of shape (1, { size }, { size }) or ({ size }, { size }) given the distribution was of shape { distrib.shape }, the shape given was { transport_plane.shape }"
		else:
			assert transport_plane.shape == (num_categories, size, size), f"The transport plane (transport_plane) passed must be of shape ({ num_categories }, { size }, { size }) given the distribution was of shape { distrib.shape }, the shape given was { transport_plane.shape }"
	if unitary_direction_matrix is not None:
		assert unitary_direction_matrix.shape == (num_dimensions, size, size), f"The unitary direction matrix (unitary_direction_matrix) passed must be of shape (num_dimensions, { size }, { size }) given the distribution was of shape { distrib.shape }, the shape given was { unitary_direction_matrix.shape }"
	# more asserts
	if local_weight_distrib is not None:
		assert local_weight_distrib.shape == (size,), f"The local weight distribution (local_weight_distrib) must be a 1-d array of length { size } given the distribution was of shape { distrib.shape }, the shape given was { local_weight_distrib.shape }"
	if category_weights is not None:
		assert not is_distrib_1dimensional, "Cannot pass a category weight vector (category_weights) if the distribution is 1-dimensional"
		assert category_weights.shape == (num_categories,), f"The lcategory weight vector (category_weights) must be a 1-d array of length { num_categories } given the distribution was of shape { distrib.shape }, the shape given was { category_weights.shape }"

	total_weight = np.sum(distrib) if category_weights is None else np.sum(category_weights)

	if local_weight_distrib is None:
		if is_null_distrib_1dimensional:
			local_weight_distrib = np.clip(null_distrib / np.sum(null_distrib), min_value_avoid_zeros, np.inf)
		else:
			local_weight_distrib = np.clip(np.sum(null_distrib, axis=0) / np.sum(null_distrib), min_value_avoid_zeros, np.inf)

	if transport_plane is None or use_same_exponent_weight:
		alpha_exponent = 1 + epsilon_exponent
		distance_mat_alpha = np.pow(distance_mat, alpha_exponent)

	if transport_plane is None:
		distrib_from, distrib_to = np.zeros_like(distrib, dtype=np.float32), np.zeros_like(null_distrib, dtype=np.float32)
		if is_null_distrib_1dimensional:
			distrib_to = null_distrib / np.sum(null_distrib)
		else:
			for category in range(num_categories):
				distrib_to[category] = null_distrib[category] / np.sum(null_distrib[category])

		if is_distrib_1dimensional:
			distrib_from = distrib / np.sum(distrib)
		else:
			for category in range(num_categories):
				distrib_from[category] = distrib[category] / np.sum(distrib[category])

		transport_plane = utils.compute_optimal_transport_flux(distrib_to, distrib_from, distance_mat_alpha, ot_emb_args=ot_emb_args, ot_emb_kwargs=ot_emb_kwargs)

	for category in range(num_categories):
		weight_this_category                  = np.sum(distrib[category]) if category_weights is None else category_weights[category]

		category_ot_result  = np.copy(transport_plane[category, :, :]) if len(transport_plane.shape) == 3 else transport_plane
		category_ot_result *= distance_mat_alpha if use_same_exponent_weight else distance_mat

		local_exiting_heterogeneity_this_category  = category_ot_result.sum(axis=0) / local_weight_distrib
		local_entering_heterogeneity_this_category = category_ot_result.sum(axis=1) / local_weight_distrib
		local_heterogeneity_this_category          = (local_exiting_heterogeneity_this_category + local_entering_heterogeneity_this_category) / 2

		if is_distrib_1dimensional:
			results.local_heterogeneity          = local_heterogeneity_this_category
			results.local_exiting_heterogeneity  = local_exiting_heterogeneity_this_category
			results.local_entering_heterogeneity = local_entering_heterogeneity_this_category
			results.local_signed_heterogeneity   = (local_exiting_heterogeneity_this_category - local_entering_heterogeneity_this_category) / 2

			results.global_heterogeneity = np.sum(local_heterogeneity_this_category * local_weight_distrib)

			if has_direction:
				for dimension in range(num_dimensions):
					results.direction[category, dimension, :] = ((unitary_direction_matrix[dimension, :, :] * category_ot_result).sum(axis=0) + (unitary_direction_matrix[dimension, :, :].T * category_ot_result).sum(axis=1)) / 2 / local_weight_distrib
		else:
			results.local_heterogeneity                                 += local_heterogeneity_this_category * weight_this_category / total_weight
			results.local_exiting_heterogeneity                         += local_exiting_heterogeneity_this_category * weight_this_category / total_weight
			results.local_entering_heterogeneity                        += local_entering_heterogeneity_this_category * weight_this_category / total_weight
			results.local_heterogeneity_per_category[category]           = local_heterogeneity_this_category
			results.local_exiting_heterogeneity_per_category[category]   = local_exiting_heterogeneity_this_category
			results.local_entering_heterogeneity_per_category[category]  = local_entering_heterogeneity_this_category
			results.local_signed_heterogeneity[category]                 = (local_exiting_heterogeneity_this_category - local_entering_heterogeneity_this_category) / 2
			
			results.global_heterogeneity_per_category[category]  = np.sum(local_heterogeneity_this_category * local_weight_distrib)
			results.global_heterogeneity                        += results.global_heterogeneity_per_category[category] * weight_this_category / total_weight

			if has_direction:
				for dimension in range(num_dimensions):
					results.direction_per_category[category, dimension, :] = ((unitary_direction_matrix[dimension, :, :] * category_ot_result).sum(axis=0) + (unitary_direction_matrix[dimension, :, :].T * category_ot_result).sum(axis=1)) / 2 / local_weight_distrib
				results.direction += results.direction_per_category[category, :, :] * weight_this_category / total_weight

	if is_distrib_1dimensional:
		distrib = distrib[0]

	if return_transport_plane:
		return results, transport_plane
	return results


def ot_heterogeneity_populations(
	distrib, distance_mat: np.array, total_population_distrib: np.array=None, unitary_direction_matrix: np.array=None,
	transport_plane: np.array=None, return_transport_plane: bool=False,
	epsilon_exponent: float=-1e-3, use_same_exponent_weight: bool=True,
	min_value_avoid_zeros: float=1e-5, ot_emb_args : list=[], ot_emb_kwargs : dict={}
):
	'''
    The ot_heterogeneity_populations function uses the total population distribution accross all classes as the null
    distribution. It thus assumes the nul distribution is the distribution where the total population at each location
    doesn't change, and the proportion of each category is the same as the global distribution of classes.

    Parameters:
        distrib (np.array): 2d-array of shape (`num_categories`, `size`) or 1d-array of length `size` representing the
        	population distribution, i.e. the population of each category in each location. A 1d array requires 
        	`total_population_distrib` to be passed.
        distance_mat (np.array): 2d-array of shape (`size`, `size`) representing the distance between each locality.
        total_population_distrib (np.array): 1d-array of length `size` representing the population at each locality,
        	usefull to compute the heterogeneity of one or multiple small group within a larger population, while
        	ignoring the majority that is outside of these small groups.
        transport_plane (np.array): either a 3d array of shape (`num_dimensions`, `size`, `size`) or a 2d array
			of shape (`size`, `size`) if distributions_from is only 1d. Element of index (n, i, j) reprensents
			the flux of population n from locality i to locality j.
		return_transport_plane (bool): if true, the function will also return the transport plane.
        unitary_direction_matrix (np.array): 3d-array of shape (`num_categories`, `size`, `size`) representing the unitary
        	vector between each location.
        epsilon_exponent (float): the distance matrix is exponentiated (element-wise) by an exponent 1+`epsilon_exponent`
        use_same_exponent_weight (bool): if true the cost (i.e. distant) is exponentiated by the same exponent as the one
        	for the cost matrix in the optimal-transport computation.
        min_value_avoid_zeros (float): value below wich a value is concidered zero.
        ot_emb_args (list): list of additional unamed argument to pass to the ot.emb function that is used as a backend.
        ot_emb_kwargs (dict): list of additional amed argument to pass to the ot.emb function that is used as a backend.

	Returns:
		results (ot_heterogeneity_results)
		transport_plane (np.array): either a 3d array of shape (`num_dimensions`, `size`, `size`) or a 2d array
			of shape (`size`, `size`) if distrib is only 1d. Element of index (n, i, j) reprensents the flux
			of population n from locality i to locality j. Returned only if return_transport_plane is true.
    '''

	is_distrib_1dimensional = len(distrib.shape) == 1

	if total_population_distrib is None:
		assert not is_distrib_1dimensional, "A reference distribution (total_population_distrib) must be passed to ot_heterogeneity_populations if the distribution (distrib) is 1-dimensional"
		null_distrib = np.sum(distrib, axis=0)
	else:
		null_distrib = total_population_distrib / np.sum(total_population_distrib) * np.sum(distrib)

	return ot_heterogeneity_from_null_distrib(
		distrib, null_distrib, distance_mat, transport_plane=transport_plane,
		return_transport_plane=return_transport_plane, unitary_direction_matrix=unitary_direction_matrix,
		epsilon_exponent=epsilon_exponent, use_same_exponent_weight=use_same_exponent_weight,
		min_value_avoid_zeros=min_value_avoid_zeros, ot_emb_args=ot_emb_args, ot_emb_kwargs=ot_emb_kwargs
	)

def ot_heterogeneity_linear_regression(
	distrib: np.array, prediction_distrib: np.array, distance_mat: np.array, local_weight_distrib: np.array=None,
	transport_plane: np.array=None, return_transport_plane: bool=False, unitary_direction_matrix: np.array=None,
	fit_regression : bool=True, regression=sklearn.linear_model.LinearRegression(), 
	epsilon_exponent: float=-1e-3, use_same_exponent_weight: bool=True,
	min_value_avoid_zeros: float=1e-5, ot_emb_args : list=[], ot_emb_kwargs : dict={}
):
	''' Will be documented later on. '''

	is_predict_distrib_1dimensional = len(prediction_distrib.shape) == 1
	is_distrib_1dimensional         = len(distrib.shape)            == 1

	num_categories = 1 if is_distrib_1dimensional else distrib.shape[0]
	size           = distrib.shape[0] if is_distrib_1dimensional else distrib.shape[1]

	if local_weight_distrib is None:
		local_weight_distrib = np.clip(np.sum(distrib, axis=0) / np.sum(distrib), min_value_avoid_zeros, np.inf)

	X_regression = np.expand_dims(prediction_distrib / local_weight_distrib, 1) if is_predict_distrib_1dimensional else (prediction_distrib / local_weight_distrib).T
	Y_regression = np.expand_dims(           distrib / local_weight_distrib, 1) if         is_distrib_1dimensional else (           distrib / local_weight_distrib).T

	if fit_regression:
		regression.fit(X_regression, Y_regression)
	null_distrib  = regression.predict(X_regression).T
	null_distrib *= local_weight_distrib / null_distrib.sum(axis=0)
	
	if is_distrib_1dimensional:
		null_distrib = null_distrib[0, :]

	return ot_heterogeneity_from_null_distrib(
		distrib, null_distrib, distance_mat,
		transport_plane=transport_plane, return_transport_plane=return_transport_plane,
		local_weight_distrib=local_weight_distrib, unitary_direction_matrix=unitary_direction_matrix,
		epsilon_exponent=epsilon_exponent, use_same_exponent_weight=use_same_exponent_weight,
		min_value_avoid_zeros=min_value_avoid_zeros, ot_emb_args=ot_emb_args, ot_emb_kwargs=ot_emb_kwargs
	), regression
