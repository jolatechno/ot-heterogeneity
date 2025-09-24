import ot
import numpy as np
from sklearn import linear_model 
from collections.abc import Iterable

class ot_heterogeneity_results:
	def __init__(self, size=0, num_categories=0, num_dimensions=1, has_direction=False):
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
				self.global_heterogeneity_per_category         = np.zeros(                 size )
				self.local_signed_heterogeneity                = np.zeros((num_categories, size))
				self.local_heterogeneity_per_category          = np.zeros((num_categories, size))
				self.local_exiting_heterogeneity_per_category  = np.zeros((num_categories, size))
				self.local_entering_heterogeneity_per_category = np.zeros((num_categories, size))
				if has_direction:
					self.direction_per_category = np.zeros((num_categories, num_dimensions, size))


def ot_heterogeneity_from_null_distrib(
	distrib, null_distrib, distance_mat,
	unitary_direction_matrix=None, local_weight_distrib=None, category_weights=None,
	alpha_exponent=None, epsilon_exponent=-1e-3,
	use_same_exponent_weight=True, min_value_avoid_zeros=1e-5
):
	if alpha_exponent is None:
		alpha_exponent = 1 + epsilon_exponent
	distance_mat_alpha = np.pow(distance_mat, alpha_exponent)

	is_local_weights_1dimensional = not isinstance(local_weight_distrib[0], Iterable) if local_weight_distrib is not None else False
	is_null_distrib_1dimensional  = not isinstance(null_distrib[0],         Iterable)
	is_distrib_1dimensional       = not isinstance(distrib[0],              Iterable)

	if local_weight_distrib is None:
		if is_null_distrib_1dimensional:
			local_weight_distrib = np.clip(null_distrib / np.sum(null_distrib), min_value_avoid_zeros, np.inf)
		else:
			local_weight_distrib = np.clip(np.sum(null_distrib, axis=0) / np.sum(null_distrib), min_value_avoid_zeros, np.inf)

	num_categories = 1 if is_distrib_1dimensional else len(distrib)
	size           = len(distrib) if is_distrib_1dimensional else len(distrib[0])
	has_direction  = unitary_direction_matrix is not None
	num_dimensions = 1 if not has_direction else len(unitary_direction_matrix)
	results        = ot_heterogeneity_results(size, num_categories, num_dimensions, has_direction)

	total_weight = np.sum(distrib) if category_weights is None else np.sum(category_weights)
	for category in range(num_categories):
		weight_this_category                  = np.sum(distrib[category]) if category_weights is None else category_weights[category]
		normalized_distrib_this_category      = distrib / np.sum(distrib) if is_distrib_1dimensional else distrib[category] / np.sum(distrib[category])
		normalized_null_distrib_this_category = null_distrib / np.sum(null_distrib) if is_null_distrib_1dimensional else null_distrib[category] / np.sum(null_distrib[category])

		category_ot_result  = ot.emd(normalized_null_distrib_this_category, normalized_distrib_this_category, distance_mat_alpha)
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

	return results


def ot_heterogeneity_populations(
	distrib, distance_mat, unitary_direction_matrix=None,
	alpha_exponent=None, epsilon_exponent=-1e-3, use_same_exponent_weight=True, 
	min_value_avoid_zeros=1e-6
):
	null_distrib = np.sum(distrib, axis=0)

	return ot_heterogeneity_from_null_distrib(
		distrib, null_distrib, distance_mat, unitary_direction_matrix=unitary_direction_matrix,
		alpha_exponent=alpha_exponent, epsilon_exponent=epsilon_exponent, use_same_exponent_weight=use_same_exponent_weight,
		min_value_avoid_zeros=min_value_avoid_zeros
	)

def ot_heterogeneity_linear_regression(
	distrib, prediction_distrib, distance_mat, local_weight_distrib=None, unitary_direction_matrix=None,
	fit_regression=True, regression=linear_model.LinearRegression(), alpha_exponent=None,
	epsilon_exponent=-1e-3, use_same_exponent_weight=True, min_value_avoid_zeros=1e-6
):
	is_predict_distrib_1dimensional = not isinstance(prediction_distrib[0], Iterable)
	is_distrib_1dimensional         = not isinstance(distrib[0],            Iterable)

	num_categories = 1 if is_distrib_1dimensional else len(distrib)
	size           = len(distrib) if is_distrib_1dimensional else len(distrib[0])

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
		distrib, null_distrib, distance_mat, local_weight_distrib=local_weight_distrib, unitary_direction_matrix=unitary_direction_matrix,
		alpha_exponent=alpha_exponent, epsilon_exponent=epsilon_exponent, use_same_exponent_weight=use_same_exponent_weight,
		min_value_avoid_zeros=min_value_avoid_zeros
	), regression
