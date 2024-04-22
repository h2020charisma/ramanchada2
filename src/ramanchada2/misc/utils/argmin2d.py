import numpy as np
import numpy.typing as npt
import pandas as pd
import pydantic
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Union


def argmin2d(A):
    ymin_idx = np.argmin(A, axis=0)
    xmin_idx = np.argmin(A, axis=1)
    x_idx = np.unique(xmin_idx[xmin_idx[ymin_idx[xmin_idx]] == xmin_idx])
    y_idx = np.unique(ymin_idx[ymin_idx[xmin_idx[ymin_idx]] == ymin_idx])
    matches = np.stack([y_idx, x_idx]).T
    return matches


def find_closest_pairs_idx(x, y):
    outer_dif = np.abs(np.subtract.outer(x, y))
    return argmin2d(outer_dif).T


def find_closest_pairs(x, y):
    x_idx, y_idx = find_closest_pairs_idx(x, y)
    return x[x_idx], y[y_idx]


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def align(x, y,
          p0: Union[List[float], npt.NDArray] = [0, 1, 0, 0],
          func=lambda x, a0, a1, a2, a3: (a0*np.ones_like(x), a1*x, a2*x**2/1, a3*(x/1000)**3),
          max_iter: pydantic.PositiveInt = 1000):
    """
    Iteratively finds best match between x and y and evaluates the x scaling parameters.
    min((lambda(x, *p)-y)**2 | *p)
    Finds best parameters *p that minimise L2 distance between scaled x and original y
    """
    if isinstance(p0, list):
        p = np.array(p0)
    else:
        p = p0
    loss = np.infty
    cur_x = x
    for it in range(max_iter):
        cur_x = np.sum(func(x, *p), axis=0)
        x_idx, y_idx = find_closest_pairs_idx(cur_x, y)
        x_match, y_match = x[x_idx], y[y_idx]
        p_bak = p
        obj_mat = np.stack(func(x_match, *np.ones_like(p)), axis=1)
        p, *_ = linalg.lstsq(obj_mat, y_match, cond=1e-8)
        loss_bak = loss
        loss = np.sum((x_match-y_match)**2)/len(x_match)**2
        if np.allclose(p, p_bak):
            break
        if loss > loss_bak:
            pass
            return p_bak
    return p


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def align_shift(x, y,
                p0: float = 0,
                max_iter: pydantic.PositiveInt = 1000):
    loss = np.infty
    cur_x = x
    p = p0
    for it in range(max_iter):
        cur_x = x + p
        x_idx, y_idx = find_closest_pairs_idx(cur_x, y)
        x_match, y_match = x[x_idx], y[y_idx]
        p_bak = p
        p = np.mean(y_match-x_match)
        loss_bak = loss
        loss = np.sum((y_match-x_match)**2)
        if np.allclose(p, p_bak):
            break
        if loss > loss_bak:
            return p_bak
    return p


def match_peaks(spe_pos_dict, ref):
    # Min-Max normalize the reference values
    min_value = min(ref.values())
    max_value = max(ref.values())
    if len(ref.keys()) > 1:
        normalized_ref = {key: (value - min_value) / (max_value - min_value) for key, value in ref.items()}
    else:
        normalized_ref = ref

    min_value_spe = min(spe_pos_dict.values())
    max_value_spe = max(spe_pos_dict.values())
    # Min-Max normalize the spe_pos_dict
    if len(spe_pos_dict.keys()) > 1:
        normalized_spe = {
                key: (value - min_value_spe) / (max_value_spe - min_value_spe) for key, value in spe_pos_dict.items()
                }
    else:
        normalized_spe = spe_pos_dict
    data_list = [
        {'Wavelength': key, 'Intensity': value, 'Source': 'spe'} for key, value in normalized_spe.items()
    ] + [
        {'Wavelength': key, 'Intensity': value, 'Source': 'reference'} for key, value in normalized_ref.items()
    ]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    feature_matrix = df[['Wavelength', 'Intensity']].to_numpy()

    n_ref = len(ref.keys())
    n_spe = len(spe_pos_dict.keys())
    kmeans = KMeans(n_clusters=n_ref if n_ref > n_spe else n_spe)
    kmeans.fit(feature_matrix)
    labels = kmeans.labels_
    # Extract cluster labels, x values, and y values
    df['Cluster'] = labels
    grouped = df.groupby('Cluster')
    x_spe = np.array([])
    x_reference = np.array([])
    x_distance = np.array([])
    clusters = np.array([])
    # Iterate through each group
    for cluster, group in grouped:
        # Get the unique sources within the group
        unique_sources = group['Source'].unique()
        if 'reference' in unique_sources and 'spe' in unique_sources:
            # Pivot the DataFrame to create the desired structure
            for w_spe in group.loc[group["Source"] == "spe"]["Wavelength"].values:
                x = None
                r = None
                e_min = None
                for w_ref in group.loc[group["Source"] == "reference"]["Wavelength"].values:
                    e = euclidean_distances(w_spe.reshape(-1, 1), w_ref.reshape(-1, 1))[0][0]
                    if (e_min is None) or (e < e_min):
                        x = w_spe
                        r = w_ref
                        e_min = e
                x_spe = np.append(x_spe, x)
                x_reference = np.append(x_reference, r)
                x_distance = np.append(x_distance, e_min)
                clusters = np.append(clusters, cluster)
    sort_indices = np.argsort(x_spe)
    return (x_spe[sort_indices], x_reference[sort_indices], x_distance[sort_indices], df)
