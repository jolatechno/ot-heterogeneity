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
		population_trajectory (np.array): trajectory of the accumulated populations categories, array of shape
			(`num_categories`, `size`, `size`)
		proportion_trajectory (np.array): proportional equivalent of the population_trajectory, array of shape
			(`num_categories`, `size`, `size`)
		divergence_trajectory (np.array): trajectory of divergence for each locality, array of shape (`size`, `size`)
		focal_time_trajectory (np.array): trajectory of focal time for each locality, array of shape (`size`, `size`)
		focal_time_trajectory_value (np.array): coresponding value to the trajectory of focal time for each locality,
			array of shape (`size`, `size`)
		distorsion_coefficient (np.array): un-normalized distorsion coefficient for each locality, array of length `size`
		distorsion_coefficient (np.array): un-normalized distorsion coefficient for each locality, array of length `size`
		normalized_distorsion_coefficient (np.array): normalized distorsion coefficient for each locality, array of length
			`size`
		normalization_coefficient (float): normalization coefficient
		normalization_population_trajectory (np.array): worst `population_trajectory` computed for normalization, array
			of shape (`num_categories`, `size`)
		normalization_proportion_trajectory (np.array): worst `proportion_trajectory` computed for normalization, array
			of shape (`num_categories`, `size`)
		normalization_divergence_trajectory (np.array): worst `divergence_trajectory` computed for normalization, 1d
			array of length `size`
		normalization_integration_trajectory (np.array): worst integration absice computed for normalization, 1d array
			of length `size`

	'''

	def __init__(self, size: int=0, num_categories: int=0):
		self.size, self.num_categories         = size, num_categories
		self.reference_distribution            = _np.zeros(num_categories)
		self.population_trajectory             = _np.zeros((num_categories, size, size))
		self.proportion_trajectory             = _np.zeros((num_categories, size, size))
		self.divergence_trajectory             = _np.zeros((size, size))
		self.focal_time_trajectory             = _np.zeros((size, size))
		self.focal_time_trajectory_value       = _np.zeros((size, size))
		self.distorsion_coefficient            = _np.zeros(size)
		self.normalized_distorsion_coefficient = _np.zeros(size)

		self.normalization_coefficient            = 0
		self.normalization_population_trajectory  = _np.zeros((num_categories, size))
		self.normalization_proportion_trajectory  = _np.zeros((num_categories, size))
		self.normalization_divergence_trajectory  = _np.zeros(size)
		self.normalization_integration_trajectory = _np.zeros(size)

def compute_multiscalar_normalization_coefficient(
	results: multiscalar_results, distance_mat: _np.array,
	divergence=divergences.KL_divergence, consider_distance: bool=False
) -> float:
	'''
	The compute_multiscalar_normalization_coefficient function computes the normalization factor for the multiscalar lens
	heterogeneity index.

	Parameters:
		results (multiscalar_results) : complete or partially complete (without normalization) results of a multiscalar
			lens heterogeneity computation.
		distance_mat (np.array): 2d-array of shape (`size`, `size`) representing the distance between each locality.

	Optional parameters:
		divergence (function (np.array, np.array) -> np.array): function to compute the divegrences, some of which
			can be found in the divergences subpackage. Default is divergences.kl_divergence
		consider_distance (bool): wether to consider the distance when computing area under the curve, default is
			false and thus reverts to using accululated population as the integration variable, as in the base
			version of the multiscalar lens index.

	Returns:
		results (multiscalar_results) with updated normalization
	'''

	size                        = results.size
	num_categories              = results.num_categories
	total_population_trajectory = _np.sum(results.population_trajectory, axis=0)
	total_population            = total_population_trajectory[0, -1]

	if consider_distance:
		results.normalization_integration_trajectory = _np.max(distance_mat,                axis=0)
		integration_variable                         = _np.diff(_np.append([0], results.normalization_integration_trajectory))
		worst_population_trajectory                  = _np.min(total_population_trajectory, axis=0)
	else:
		integration_variable                         = _np.full(size, total_population/size)
		worst_population_trajectory                  = _np.cumsum(integration_variable)
		results.normalization_integration_trajectory = worst_population_trajectory

	reference_distribution  = _np.sort(results.reference_distribution)
	population_distribution = reference_distribution*total_population

	results.normalization_population_trajectory = _np.zeros((num_categories, size))
	running_added_population = _np.zeros(num_categories)
	for step in range(size):
		population_to_add = worst_population_trajectory[step] - (0 if step == 0 else worst_population_trajectory[step-1])
		population_not_added_yet = population_distribution - running_added_population

		for category in range(num_categories):
			results.normalization_population_trajectory[category, step] = min(population_to_add, population_not_added_yet[category])
			population_to_add -= results.normalization_population_trajectory[category, step]
			if population_to_add <= 0:
				break

		running_added_population += results.normalization_population_trajectory[:, step]
	results.normalization_population_trajectory = _np.cumsum(results.normalization_population_trajectory, axis=1)

	results.normalization_proportion_trajectory = _np.zeros_like(results.normalization_population_trajectory)
	for category in range(num_categories):
		results.normalization_proportion_trajectory[category, :] = results.normalization_population_trajectory[category, :] / worst_population_trajectory

	results.normalization_divergence_trajectory = divergence(results.normalization_proportion_trajectory, reference_distribution)

	results.normalization_coefficient         = _np.sum(integration_variable * results.normalization_divergence_trajectory)
	results.normalized_distorsion_coefficient = results.distorsion_coefficient / results.normalization_coefficient

	return results


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
		divergence (function (np.array, np.array) -> np.array): function to compute the divegrences, some of which
			can be found in the divergences subpackage. Default is divergences.kl_divergence
		consider_distance (bool): wether to consider the distance when computing area under the curve, default is
			false and thus reverts to using accululated population as the integration variable, as in the base
			version of the multiscalar lens index.
		divergence_kwargs (dict): list of additional argument to pass to the divegrence function.

	Returns:
		results (multiscalar_results)
	'''

	is_distrib_1dimensional = len(distrib.shape) == 1
	num_categories          = 1 if is_distrib_1dimensional else distrib.shape[0]
	size                    = distrib.shape[0] if is_distrib_1dimensional else distrib.shape[1]
	results                 = multiscalar_results(size, num_categories)

	for category in range(num_categories):
		results.population_trajectory[category, :, :] = _utils.compute_cumulative_neighbor_cost(distance_mat, distrib[category, :] if not is_distrib_1dimensional else distrib, conserve_order=False)
	total_population_trajectory = _np.sum(results.population_trajectory, axis=0)
	for category in range(num_categories):
		results.proportion_trajectory[category, :, :] = results.population_trajectory[category, :, :] / total_population_trajectory

	results.reference_distribution = results.proportion_trajectory[:, 0, -1] if total_population_distrib is None else total_population_distrib/_np.sum(total_population_distrib)
	distance_mat_  = _np.sort(distance_mat, axis=1) if consider_distance else total_population_trajectory

	results.divergence_trajectory = divergence(results.proportion_trajectory.reshape(num_categories, size * size), results.reference_distribution).reshape(size, size)

	armonized_divergence_trajectory_index = _np.full_like(results.divergence_trajectory, size-1, dtype=int)
	for step in range(size-1, 0, -1):
		is_greater = results.divergence_trajectory[:, step-1] > results.divergence_trajectory[_np.arange(size), armonized_divergence_trajectory_index[:, step]]
		armonized_divergence_trajectory_index[is_greater, 0:step] = step-1

	index_mat = _np.repeat(_np.expand_dims(_np.arange(size), axis=0), size, axis=0)
	results.focal_time_trajectory       = distance_mat_[                index_mat.T, armonized_divergence_trajectory_index]
	results.focal_time_trajectory_value = results.divergence_trajectory[index_mat.T, armonized_divergence_trajectory_index]

	integration_variable           = _np.diff(_np.append(_np.zeros((size, 1)), results.focal_time_trajectory, axis=1), axis=1)
	results.distorsion_coefficient = _np.sum(integration_variable * results.focal_time_trajectory_value, axis=1)

	results = compute_multiscalar_normalization_coefficient(results, distance_mat_, divergence, consider_distance)

	return results