import numpy as _np

from .. import utils as _utils
from . import _multiscalar_divergences as divergences

class multiscalar_results:
	'''
    The multiscalar_results class contains all of the results of a computation of spatial heterogeneity based on the
    multiscalar lens method.

    Attributes:
        size (int): Number of spatial units (town, polling stations, etc...)
        num_categories (int): number of distinct categories
        reference_distribution (np.array): 1d array of length `num_categories` representing the reference distibution
        	for the total population.
        divergence_trajectory (np.array): trajectory of divergence for each locality, array of shape (`size`, `size`)
        focal_time_trajectory (np.array): trajectory of focal time for each locality, array of shape (`size`, `size`)
        distorsion_coefficient (np.array): un-normalized distorsion coefficient for each locality, array of length `size`
        distorsion_coefficient (np.array): un-normalized distorsion coefficient for each locality, array of length `size`
        normalized_distorsion_coefficient (np.array): normalized distorsion coefficient for each locality, array of length `size`
        normalization_coefficient (float): normalization coefficient

    '''

	def __init__(self, size: int=0, num_categories: int=0):
		self.size, self.num_categories         = size, num_categories
		self.reference_distribution            = _np.zeros(num_categories)
		self.divergence_trajectory             = _np.zeros((size, size))
		self.focal_time_trajectory             = _np.zeros((size, size))
		self.distorsion_coefficient            = _np.zeros(size)
		self.normalized_distorsion_coefficient = _np.zeros(size)
		self.normalization_coefficient         = 0

def compute_multiscalar_normalization_coefficient(
	distrib : _np.array, distance_mat: _np.array, total_population_distrib: _np.array=None,
	divergence=divergences.KL_divergence, consider_distance: bool=False, divergence_kwargs: dist={}
) -> float:
	'''
    The compute_multiscalar_normalization_coefficient function computes the normalization factor for the multiscalar lens
    heterogeneity index.

    Parameters:
        distrib (np.array): 2d-array of shape (`num_categories`, `size`) or 1d-array of length `size` representing the
        	population distribution, i.e. the population of each category in each location. A 1d array requires 
        	`total_population_distrib` to be passed.
        distance_mat (np.array): 2d-array of shape (`size`, `size`) representing the distance between each locality.

    Optional parameters:
        total_population_distrib (np.array): 1d-array of length `size` representing the population at each locality,
        	usefull to compute the heterogeneity of one or multiple small group within a larger population, while
        	ignoring the majority that is outside of these small groups.
        min_value_avoid_zeros (float): value below wich a value is concidered zero.
        divergence (function (np.array, np.array) -> np.array): function to compute the divegrences, some of which
        	can be found in the divergences subpackage. Default is divergences.kl_divergence
        consider_distance (bool): wether to consider the distance when computing area under the curve, default is
        	false as in the base version of the multiscalar lens index.
        divergence_kwargs (dict): list of additional argument to pass to the divegrence function.

	Returns:
		normalization_coefficient (float)
    '''

	null_distrib  = _np.sum(distrib, axis=0)/_np.sum(distrib) if total_population_distrib is None else total_population_distrib/_np.sum(total_population_distrib)
	distance_mat_ = distance_mat if consider_distance else _utils.compute_neighbor_index_matrix(distance_mat)

	# TODO

	return 1.0

def compute_multiscalar(
	distrib : _np.array, distance_mat: _np.array, total_population_distrib: _np.array=None,
	divergence=divergences.KL_divergence, consider_distance: bool=False, divergence_kwargs: dist={}
) -> multiscalar_results:
	'''
    The compute_multiscalar function computes the multiscalar lens heterogeneity index...

    Parameters:
        distrib (np.array): 2d-array of shape (`num_categories`, `size`) or 1d-array of length `size` representing the
        	population distribution, i.e. the population of each category in each location. A 1d array requires 
        	`total_population_distrib` to be passed.
        distance_mat (np.array): 2d-array of shape (`size`, `size`) representing the distance between each locality.

    Optional parameters:
        total_population_distrib (np.array): 1d-array of length `size` representing the population at each locality,
        	usefull to compute the heterogeneity of one or multiple small group within a larger population, while
        	ignoring the majority that is outside of these small groups.
        min_value_avoid_zeros (float): value below wich a value is concidered zero.
        divergence (function (np.array, np.array) -> np.array): function to compute the divegrences, some of which
        	can be found in the divergences subpackage. Default is divergences.kl_divergence
        consider_distance (bool): wether to consider the distance when computing area under the curve, default is
        	false as in the base version of the multiscalar lens index.
        divergence_kwargs (dict): list of additional argument to pass to the divegrence function.

	Returns:
		results (multiscalar_results)
    '''

	is_distrib_1dimensional = len(distrib.shape) == 1
	num_categories          = 1 if is_distrib_1dimensional else distrib.shape[0]
	size                    = distrib.shape[0] if is_distrib_1dimensional else distrib.shape[1]
	results                 = multiscalar_results(size, num_categories)
	null_distrib            = _np.sum(distrib, axis=0)/_np.sum(distrib) if total_population_distrib is None else total_population_distrib/_np.sum(total_population_distrib)
	distance_mat_           = distance_mat if consider_distance else _utils.compute_neighbor_index_matrix(distance_mat)

	results.reference_distribution = null_distrib

    # TODO

	results.normalization_coefficient = compute_multiscalar_normalization_coefficient(distrib, distance_mat, total_population_distrib, divergence, consider_distance, divergence_kwargs)
	results.normalized_distorsion_coefficient = results.distorsion_coefficient / results.normalization_coefficient

	return results
