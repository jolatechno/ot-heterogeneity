# ot-heterogeneity

A project to compute optimal transport based heterogeneity indexes.

## 1 - Usage

The librairy can simply be installed using `pip install oterogeneity` and then imported and used as documented here :

```python
import oterogeneity as oth
from oterogeneity import utils

unitary_direction_matrix, distance = utils.compute_unitary_direction_matrix_polar(lat, lon)
results = oth.ot_heterogeneity_populations(
	distrib_canidates, distance_matrix, unitary_direction_matrix
)
```

### 1.a - The result class

The `ot_heterogeneity_results` class contains all of the results of a computation of spatial heterogeneity based on optimal transport using our method.

It contains the following attributes (that may be `None` if not applicable) :
 - `size` (_`int`_): Number of spatial units (town, polling stations, etc...)
 - `num_categories` (_`int`_): number of distinct categories
 - `num_dimensions` (_`int`_): number of spacial dimensions (tympically 2)
 - `has_direction` (_`bool`_): whether the result contains directionality fields or not
 - `global_heterogeneity` (_`float`_): global heterogeneity index
 - `global_heterogeneity_per_category` (_`np.array`_): 1d-array of length `num_categories` that contains the local heterogeneity index for each category.
 - `local_heterogeneity` (_`np.array`_): 1d-array of length `size` that contains the local heterogeneity index for each location
 - `local_signed_heterogeneity` (_`np.array`_): either a 2d-array of shape (`num_categories`, `size`) when `num_categories` > 1, or a 1d-array of length `size` if `num_categories` = 1, that contains the signed heterogeneity index for each category and each location.
 - `local_exiting_heterogeneity` (_`np.array`_): 1d-array of length `size` that contains the heterogeneity index based only on exiting flux for each location.
 - `local_entering_heterogeneity` (_`np.array`_): 1d-array of length `size` that contains the heterogeneity index based only on entering flux for each location.
 - `local_heterogeneity_per_category` (_`np.array`_): 1d-array of length `size` that contains the heterogeneity index for each location.
 - `local_exiting_heterogeneity_per_category` (_`np.array`_): 2d-array of shape (`num_categories`, `size`) that contains the heterogeneity index based only on exiting flux for each category and each location.
 - `local_entering_heterogeneity_per_category` (_`np.array`_): 2d-array of shape (`num_categories`, `size`) that contains the heterogeneity index based only on entering flux for each category and each location.
 - `direction` (_`np.array`_): 2d-array of shape (`num_dimensions`, `size`) representing the vectorial field of directionality.
 - `direction_per_category` (_`np.array`_): 3d-array of shape (`num_categories`, `num_dimensions`, `size`) representing the vectorial field of directionality for each category.

### 1.b - Functions

#### 1.b.1 - `ot_heterogeneity_from_null_distrib`

The `ot_heterogeneity_from_null_distrib` function is the most general function implementing our method for measuring spatial heterogeneity.

```python
def ot_heterogeneity_from_null_distrib(
	distrib, null_distrib, distance_mat,
	unitary_direction_matrix=None, local_weight_distrib=None, category_weights=None,
	epsilon_exponent=-1e-3 : float, use_same_exponent_weight=True : bool,
	min_value_avoid_zeros=1e-5 : float
)
```

The following parameters are passed to the function :
 - `distrib` (_`np.array`_): 2d-array of shape (`num_categories`, `size`) representing the population distribution, i.e. the population of each category in each location. 
 - `null_distrib` (_`np.array`_): either a 2d-array of shape (`num_categories`, `size`) or a 1d-array of length `size` if every category has the same null distribution, representing the null distribution (distribution without heterogeneity), to which the distribution will be compared.
 - `distance_mat` (_`np.array`_): 2d-array of shape (`size`, `size`) representing the distance between each locality.

With the following parameters being optional :
 - `unitary_direction_matrix` (_`np.array`_): 3d-array of shape (`num_categories`, `size`, `size`) representing the unitary vector between each location.
 - `local_weight_distrib` (_`np.array`_): 1d-array of length `size` representing the weight for each location. By default this weight is simply the proportion of the total population located in each location.
 - `category_weights` (_`np.array`_): 1d-array of length `num_categories` representing the weight for each num_category. By default this weight is simply the proportion of the total population that belong to each category.
 - `epsilon_exponent` (_`float`_): the distance matrix is exponentiated (element-wise) by an exponent `1+epsilon_exponent`
 - `use_same_exponent_weight` (_`bool`_): if true the cost (i.e. distant) is exponentiated by the same exponent as the one for the cost matrix in the optimal-transport computation.
 - `min_value_avoid_zeros` (_`float`_): value below wich a value is concidered zero.
 - `ot_emb_args` (_`list`_): list of additional unamed argument to pass to the `ot.emb` function that is used as a backend.
 - `ot_emb_kwargs` (_`dict`_): list of additional amed argument to pass to the `ot.emb` function that is used as a backend.

The function returns a result as an object of class `ot_heterogeneity_results`.

#### 1.b.2 - `ot_heterogeneity_populations`

The `ot_heterogeneity_populations` function uses the total population distribution accross all classes as the null distribution. It thus assumes the nul distribution is the distribution where the total population at each location doesn't change, and the proportion of each category is the same as the global distribution of classes.

```python
def ot_heterogeneity_from_null_distrib(
	distrib, null_distrib, distance_mat,
	unitary_direction_matrix=None, local_weight_distrib=None, category_weights=None,
	epsilon_exponent: float=-1e-3, use_same_exponent_weight: bool=True,
	min_value_avoid_zeros: float=1e-5
)
```

The following parameters are passed to the function :
 - `distrib` (_`np.array`_): 2d-array of shape (`num_categories`, `size`) representing the population distribution, i.e. the population of each category in each location.
 - `distance_mat` (_`np.array`_): 2d-array of shape (`size`, `size`) representing the distance between each locality.

With the following parameters being optional :
 - `unitary_direction_matrix` (_`np.array`_): 3d-array of shape (`num_categories`, `size`, `size`) representing the unitary vector between each location.
 - `epsilon_exponent` (_`float`_): the distance matrix is exponentiated (element-wise) by an exponent `1+epsilon_exponent`
 - `use_same_exponent_weight` (_`bool`_): if true the cost (i.e. distant) is exponentiated by the same exponent as the one for the cost matrix in the optimal-transport computation.
 - `min_value_avoid_zeros` (_`float`_): value below wich a value is concidered zero.
 - `ot_emb_args` (_`list`_): list of additional unamed argument to pass to the `ot.emb` function that is used as a backend.
 - `ot_emb_kwargs` (_`dict`_): list of additional amed argument to pass to the `ot.emb` function that is used as a backend.

The function returns a result as an object of class `ot_heterogeneity_results`.

#### 1.b.1 - `ot_heterogeneity_linear_regression`

_The `ot_heterogeneity_linear_regression` function will be documented later on._

```python
def ot_heterogeneity_linear_regression(
	distrib, prediction_distrib, distance_mat, local_weight_distrib=None, unitary_direction_matrix=None,
	fit_regression : bool=True, regression=linear_model.LinearRegression(), 
	epsilon_exponent: float=-1e-3, use_same_exponent_weight: bool=True,
	min_value_avoid_zeros: float=1e-5
)
```

### 1.c - Utility functions

#### 1.c.1 - `compute_distance_matrix`

The `compute_distance_matrix` function computes the distance between a list of coordinates.

```python
def compute_distance_matrix(coordinates, exponent: float=2)
```

The `compute_distance_matrix` function takes the following parameters :
 - `coordinates` (_`np.array`_): 2d-array of shape (`num_dimensions`, `size`) representing the position of each location.
 - `exponent` (_`float`_): the exponent used in the norm (2 is the euclidien norm).

It returns the distance matrix filled with the distance between each location.

#### 1.c.2 - `compute_distance_matrix_polar`

The `compute_distance_matrix_polar` function computes the distance between a list of coordinates from polar coordinates on a sphere. by default it can be used for typical coordinates on earth.

```python
def compute_distance_matrix_polar(latitudes, longitudes, radius: float=6378137, unit: str="deg")
```

The `compute_distance_matrix` function takes the following parameters :
 - `latitudes` (_`np.array`_): 1d-array of length `size` with the latitudes of each point.
 - `longitudes` (_`np.array`_): 1d-array of length `size` with the longitudes of each point.
 - `radius` (_`float`_): radius of the sphere (by default 6378137 which is the radius of the earth in meters).
 - `unit` (_`str`_): a string to define the unit of the longitude and latituden, eather "rad", "deg" (default), "arcmin", or "arcsec".

It returns the distance matrix filled with the distance between each location.

#### 1.c.3 - `compute_unitary_direction_matrix`

The `compute_unitary_direction_matrix` function computes the matrix of unitary vectors used to computed direction in the main functions.

```python
def compute_unitary_direction_matrix(coordinates, distance_mat=None, exponent: float=2)
```

The `compute_unitary_direction_matrix` function takes the following parameters :
 - `coordinates` (_`np.array`_): 2d-array of shape (`num_dimensions`, `size`) representing the position of each location.
 - `distance_mat` (_`np.array`_): you can optionally pass a 2d-array of shape (`size`, `size`) filled with the distance between each location. If not passed it will be computed and returned.
 - `exponent` (_`float`_): the exponent used in the norm (2 is the euclidien norm). If a distance matrix is passed, it must have been computed with the same exponent as the one passed to this function.
	
It returns the following values :
 - `unitary_direction_matrix` (_`np.array`_): 3d-array of shape (`num_categories`, `size`, `size`) representing the unitary vector between each location.
 - `distance_mat` (_`np.array`_): a distance matrix is returned if it was not passed as a parameter (to avoid recomputing it), it is a 2d-array of shape (`size`, `size`) filled with the distance between each location.

#### 1.c.4 - `compute_unitary_direction_matrix_polar`

The `compute_unitary_direction_matrix_polar` function computes the matrix of unitary vectors used to computed direction in the main functions, between a list of coordinates from polar coordinates on a sphere. by default it can be used for typical coordinates on earth.

```python
def compute_unitary_direction_matrix_polar(latitudes, longitudes, distance_mat=None, radius: float=6378137, unit: str="deg")
```

The `compute_unitary_direction_matrix_polar` function takes the following parameters :
 - `latitudes` (_`np.array`_): 1d-array of length `size` with the latitudes of each point.
 - `longitudes` (_`np.array`_): 1d-array of length `size` with the longitudes of each point.
 - `radius` (_`float`_): radius of the sphere (by default 6378137 which is the radius of the earth in meters).
 - `distance_mat` (_`np.array`_): you can optionally pass a 2d-array of shape (`size`, `size`) filled with the distance between each location. If not passed it will be computed and returned.
 - `unit` (_`str`_): a string to define the unit of the longitude and latituden, eather "rad", "deg" (default), "arcmin", or "arcsec".

It returns the following values :
 - `unitary_direction_matrix` (_`np.array`_): 3d-array of shape (`num_categories`, `size`, `size`) representing the unitary vector between each location.
 - `distance_mat` (_`np.array`_): a distance matrix is returned if it was not passed as a parameter (to avoid recomputing it), it is a 2d-array of shape (`size`, `size`) filled with the distance between each location.

## 2 - License

```
"oterogeneity" (c) by @jolatechno - Joseph Touzet

"oterogeneity" is licensed under a
Creative Commons Attribution 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by/4.0/>.
```