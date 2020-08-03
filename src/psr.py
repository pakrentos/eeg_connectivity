from operator import sub
import numpy as np
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from toolz import curry
import warnings
warnings.filterwarnings('ignore')

# параметры для альфа частот (8-12 гц): 6 lag, 5 dims
# для теты (4-8 гц): 11 lag, 5 dims
def reconstruct(x, lag, n_dims):
    x = _vector(x)

    if lag * (n_dims - 1) >= x.shape[0] // 2:
        raise ValueError('longest lag cannot be longer than half the length of x(t)')

    lags = lag * np.arange(n_dims)
    return np.vstack(x[lag:lag - lags[-1] or None] for lag in lags).transpose()


def global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=10, **cutoffs):
    dimensions = np.arange(min_dims, max_dims + 1)
    false_neighbor_pcts = np.array([_gfnn(x, lag, n_dims, **cutoffs) for n_dims in dimensions])
    return dimensions, false_neighbor_pcts


def _gfnn(x, lag, n_dims, **cutoffs):
    offset = lag*n_dims
    is_true_neighbor = _is_true_neighbor(x, _radius(x), offset)
    return np.mean([
        not is_true_neighbor(indices, distance, **cutoffs)
        for indices, distance in _nearest_neighbors(reconstruct(x, lag, n_dims))
        if (indices + offset < x.size).all()
    ])


def _radius(x):
    return np.sqrt(((x - x.mean())**2).mean())


@curry
def _is_true_neighbor(
        x, attractor_radius, offset, indices, distance,
        relative_distance_cutoff=15,
        relative_radius_cutoff=2
):
    distance_increase = np.abs(sub(*x[indices + offset]))
    return (distance_increase / distance < relative_distance_cutoff and
            distance_increase / attractor_radius < relative_radius_cutoff)


def _nearest_neighbors(y):
    distances, indices = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(y).kneighbors(y)
    for distance, index in zip(distances, indices):
        yield index, distance[1]


def ami(x, y=None, n_bins=10):
    x, y = _vector_pair(x, y)
    if x.shape[0] != y.shape[0]:
        raise ValueError('timeseries must have the same length')
    return metrics.mutual_info_score(None, None, contingency=np.histogram2d(x, y, bins=n_bins)[0])


def lagged_ami(x, min_lag=0, max_lag=None, lag_step=1, n_bins=10):
    if max_lag is None:
        max_lag = x.shape[0]//2
    lags = np.arange(min_lag, max_lag, lag_step)

    amis = [ami(reconstruct(x, lag, 2), n_bins=n_bins) for lag in lags]
    return lags, np.array(amis)


def _vector_pair(a, b):
    a = np.squeeze(a)
    if b is None:
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError('with one input, array must have be 2D with two columns')
        a, b = a[:, 0], a[:, 1]
    return a, np.squeeze(b)


def _vector(x):
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError('x(t) must be a 1-dimensional signal')
    return x


def determine_coefs(x, min_lag=0, max_lag=20, min_dims=1, max_dims=10):
    lag_i, lag_d = lagged_ami(x, min_lag=min_lag, max_lag=max_lag, n_bins=10)
    lag = lag_i[np.argmin(lag_d)]
    dim_i, dim_d = global_false_nearest_neighbors(x, lag, max_dims=max_dims, min_dims=min_dims)
    ind = 0
    for i in range(len(dim_d)):
        if dim_d[i] == 0:
            ind = i
            break
    dim = dim_i[ind+1]
    return lag, dim
