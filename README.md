# ot-heterogeneity

A project to compute optimal transport based heterogeneity indexes.

## Usage

### The result class

The `ot_heterogeneity_results` class contains all of the results of a computation of spatial heterogeneity based on optimal transport using our method.

It contains the following attributes (that may be `None` if not applicable) :
 - `size` (_`int`_): Number of spatial units (town, polling stations, etc...)
 - `num_categories` (_`int`_): number of distinct categories
 - `num_dimensions` (_`int`_): number of spacial dimensions (tympically 2)
 - `has_direction` (_`bool`_): whether the result contains directionality fields or not
 - `global_heterogeneity` (_`float`_): global heterogeneity index
 - `global_heterogeneity_per_category` (_`np.array`_): 1d array of length `num_categories` that contains the local heterogeneity index for each category.
 - `local_heterogeneity` (_`np.array`_): 1d array of length `size` that contains the local heterogeneity index for each location
 - `local_signed_heterogeneity` (_`np.array`_): either a 2d-array of shape (`num_categories`, `size`) when `num_categories` > 1, or a 1d array of length `size` if `num_categories` = 1, that contains the signed heterogeneity index for each category and each location.
 - `local_exiting_heterogeneity` (_`np.array`_): 1d-array of length `size` that contains the heterogeneity index based only on exiting flux for each location.
 - `local_entering_heterogeneity` (_`np.array`_): 1d-array of length `size` that contains the heterogeneity index based only on entering flux for each location.
 - `local_heterogeneity_per_category` (_`np.array`_): 1d-array of length `size` that contains the heterogeneity index for each location.
 - `local_exiting_heterogeneity_per_category` (_`np.array`_): 2d-array of shape (`num_categories`, `size`) that contains the heterogeneity index based only on exiting flux for each category and each location.
 - `local_entering_heterogeneity_per_category` (_`np.array`_): 2d-array of shape (`num_categories`, `size`) that contains the heterogeneity index based only on entering flux for each category and each location.
 - `direction` (_`np.array`_): 2d-array of shape (`num_dimensions`, `size`) representing the vectorial field of directionality.
 - `direction_per_category` (_`np.array`_): 3d-array of shape (`num_categories`, `num_dimensions`, `size`) representing the vectorial field of directionality for each category.

### Functions

```python
def ot_heterogeneity_from_null_distrib(
	distrib, null_distrib, distance_mat,
	unitary_direction_matrix=None, local_weight_distrib=None, category_weights=None,
	epsilon_exponent=-1e-3 : float, use_same_exponent_weight=True : bool,
	min_value_avoid_zeros=1e-5 : float
)
```

```python
def ot_heterogeneity_populations(
	distrib, distance_mat, unitary_direction_matrix=None,
	epsilon_exponent=-1e-3 : float, use_same_exponent_weight=True : bool, 
	min_value_avoid_zeros=1e-6 : float
)
```

```python
def ot_heterogeneity_linear_regression(
	distrib, prediction_distrib, distance_mat, local_weight_distrib=None, unitary_direction_matrix=None,
	fit_regression=True : bool, regression=linear_model.LinearRegression(), epsilon_exponent=-1e-3 : float,
	use_same_exponent_weight=True : bool, min_value_avoid_zeros=1e-6 : float
)
```

### Utility functions

def compute_distance_matrix(coordinates, exponent=2 : float)

def compute_distance_matrix_polar(latitudes, longitudes, radius=6378137  : float, unit="deg" : str)

def compute_unitary_direction_matrix(coordinates, distance_mat=None, exponent=2 : float)

def compute_unitary_direction_matrix_polar(latitudes, longitudes, distance_mat=None, radius=6378137 : float, unit="deg" : str)


## License

```
"oterogeneity" (c) by @jolatechno - Joseph Touzet

"oterogeneity" is licensed under a
Creative Commons Attribution 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by/4.0/>.
```