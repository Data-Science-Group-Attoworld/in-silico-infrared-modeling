import numpy as np
import pandas as pd

# Import dependencies for Nested_CV_Parameter_Opt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc

from sklearn.neighbors import NearestNeighbors
import torch

from scipy.linalg import sqrtm

from itertools import groupby



def peak_ratio(df):
    df_peak_ratios = pd.DataFrame()
    df_peak_ratios['I_1635/I_1654'] = df[1635.35070800781]/df[1654.63562011719]
    df_peak_ratios['I_1546/I_1655'] = df[1546.64074707031]/df[1654.63562011719]
    df_peak_ratios['I_1655/(I_1655+I_1548)'] = df[1654.63562011719]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1684/(I_1655+I_1548)'] = df[1683.56274414062]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1515/(I_1655+I_1548)'] = df[1515.78503417969]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_2959/I_2931'] = df[2958.28784179687]/df[2931.2890625]
    df_peak_ratios['(I_2855+I_2927)/(I_2962+I_2871)'] = (df[2854.14990234375]+df[2927.43212890625])/(df[2962.14477539063]+df[2871.50634765625])
    df_peak_ratios['(I_2851+I_2927)/(I_1655+I_1548)'] = (df[2850.29296875]+df[2927.43212890625])/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1239/(I_2851+I_2927)'] = df[1238.083984375]/(df[2850.29296875]+df[2927.43212890625])
    df_peak_ratios['I_1741/I_1640'] = df[1741.41711425781]/df[1639.20776367187]
    df_peak_ratios['I_1740/I_1400'] = df[1739.48864746094]/df[1400.07629394531]
    df_peak_ratios['I_2852/I_1400'] = df[2852.22143554687]/df[1400.07629394531]
    df_peak_ratios['I_1450/I_1539'] = df[1450.21667480469]/df[1538.9267578125]
    df_peak_ratios['I_1240/I_1517'] = df[1240.01245117188]/df[1517.71350097656]
    df_peak_ratios['I_1045/I_1545'] = df[1045.23596191406]/df[1544.71228027344]
    df_peak_ratios['I_1080/I_1550'] = df[1079.94860839844]/df[1550.49768066406]
    df_peak_ratios['I_1060/I_1230'] = df[1060.66381835938]/df[1230.36999511719]
    df_peak_ratios['I_1170/I_1080'] = df[1170.58715820312]/df[1079.94860839844]
    df_peak_ratios['I_1030/I_1080'] = df[1029.80810546875]/df[1079.94860839844]
    df_peak_ratios['I_1080/I_1243'] = df[1079.94860839844]/df[1243.86938476562]
    df_peak_ratios['I_1587/(I_1655+I_1548)'] = df[1587.13879394531]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1156/I_1171'] = df[1155.15930175781]/df[1170.58715820312]
    df_peak_ratios['I_1243/I_1314'] = df[1243.86938476562]/df[1313.29467773438]
    df_peak_ratios['I_1453/I_1400'] = df[1452.14514160156]/df[1400.07629394531]
    return df_peak_ratios


def peak_ratio_numpy(data, vec):
    # Map column names to indices for easier access
    col_indices = {col: i for i, col in enumerate(vec)}

    # List of calculated ratios and their corresponding names
    ratios = []
    new_vec = []

    # Helper function to fetch columns based on the mapping
    def get_col(col_name):
        return data[:, col_indices[col_name]]

    # Add ratio calculations to the list
    ratios.append(get_col(1635.35070800781) / get_col(1654.63562011719))
    new_vec.append('I_1635/I_1654')

    ratios.append(get_col(1546.64074707031) / get_col(1654.63562011719))
    new_vec.append('I_1546/I_1655')

    ratios.append(get_col(1654.63562011719) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1655/(I_1655+I_1548)')

    ratios.append(get_col(1683.56274414062) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1684/(I_1655+I_1548)')

    ratios.append(get_col(1515.78503417969) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1515/(I_1655+I_1548)')

    ratios.append(get_col(2958.28784179687) / get_col(2931.2890625))
    new_vec.append('I_2959/I_2931')

    ratios.append((get_col(2854.14990234375) + get_col(2927.43212890625)) / (get_col(2962.14477539063) + get_col(2871.50634765625)))
    new_vec.append('(I_2855+I_2927)/(I_2962+I_2871)')

    ratios.append((get_col(2850.29296875) + get_col(2927.43212890625)) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('(I_2851+I_2927)/(I_1655+I_1548)')

    ratios.append(get_col(1238.083984375) / (get_col(2850.29296875) + get_col(2927.43212890625)))
    new_vec.append('I_1239/(I_2851+I_2927)')

    ratios.append(get_col(1741.41711425781) / get_col(1639.20776367187))
    new_vec.append('I_1741/I_1640')

    ratios.append(get_col(1739.48864746094) / get_col(1400.07629394531))
    new_vec.append('I_1740/I_1400')

    ratios.append(get_col(2852.22143554687) / get_col(1400.07629394531))
    new_vec.append('I_2852/I_1400')

    ratios.append(get_col(1450.21667480469) / get_col(1538.9267578125))
    new_vec.append('I_1450/I_1539')

    ratios.append(get_col(1240.01245117188) / get_col(1517.71350097656))
    new_vec.append('I_1240/I_1517')

    ratios.append(get_col(1045.23596191406) / get_col(1544.71228027344))
    new_vec.append('I_1045/I_1545')

    ratios.append(get_col(1079.94860839844) / get_col(1550.49768066406))
    new_vec.append('I_1080/I_1550')

    ratios.append(get_col(1060.66381835938) / get_col(1230.36999511719))
    new_vec.append('I_1060/I_1230')

    ratios.append(get_col(1170.58715820312) / get_col(1079.94860839844))
    new_vec.append('I_1170/I_1080')

    ratios.append(get_col(1029.80810546875) / get_col(1079.94860839844))
    new_vec.append('I_1030/I_1080')

    ratios.append(get_col(1079.94860839844) / get_col(1243.86938476562))
    new_vec.append('I_1080/I_1243')

    ratios.append(get_col(1587.13879394531) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1587/(I_1655+I_1548)')

    ratios.append(get_col(1155.15930175781) / get_col(1170.58715820312))
    new_vec.append('I_1156/I_1171')

    ratios.append(get_col(1243.86938476562) / get_col(1313.29467773438))
    new_vec.append('I_1243/I_1314')

    ratios.append(get_col(1452.14514160156) / get_col(1400.07629394531))
    new_vec.append('I_1453/I_1400')

    # Stack all calculated ratios as columns in a new array
    result = np.column_stack(ratios)

    return result, new_vec


def standardized_mean_difference(data_0, data_1):
    mean_0, mean_1 = np.mean(data_0, axis=0), np.mean(data_1, axis=0)
    var_0, var_1 = np.var(data_0, axis=0), np.var(data_1, axis=0)
    diff_fp = mean_0 - mean_1
    std_diff_fp = np.sqrt(var_0)
    effect_size = diff_fp/std_diff_fp
    return effect_size


def cohens_d(data_0, data_1):
    data_0 = data_0.astype(float)
    data_1 = data_1.astype(float)
    n1, n2 = np.shape(data_0)[0], np.shape(data_1)[0]
    diff_fp = np.mean(data_0, axis=0) - np.mean(data_1, axis=0)
    var_0, var_1 = np.var(data_0, axis=0), np.var(data_1, axis=0)
    pooled_std = np.sqrt(((n1 - 1) * var_0 + (n2 - 1) * var_1) / (n1 + n2 - 2))
    effect_size = diff_fp/pooled_std
    return effect_size



def rbf_kernel(X, Y, gamma):
    """
    Compute RBF kernel matrix between X and Y.
    
    Parameters:
        X: (n, d)
        Y: (m, d)
        gamma: float (1 / (2 * sigma^2))
        
    Returns:
        K: (n, m) kernel matrix
    """
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    dist_sq = X_norm + Y_norm - 2 * X @ Y.T
    return np.exp(-gamma * dist_sq)


def compute_mmd(X, Y, gamma=None, n_pc_components=10):
    """
    Compute unbiased MMD^2 between two datasets using RBF kernel.
    
    Parameters:
        X: array (n, d)
        Y: array (m, d)
        gamma: float or None
            If None, uses median heuristic.
    
    Returns:
        mmd_squared: float
    """
    X = X.copy()
    Y = Y.copy()
    
    if n_pc_components is not None:
        pca = PCA(n_components=n_pc_components)
        X = pca.fit_transform(X)
        Y = pca.transform(Y)
    
    n = X.shape[0]
    m = Y.shape[0]
    
    # Median heuristic if gamma not provided
    if gamma is None:
        Z = np.vstack([X, Y])
        dists = np.sum((Z[:, None, :] - Z[None, :, :])**2, axis=2)
        median_dist = np.median(dists)
        gamma = 1.0 / (2 * median_dist + 1e-12)
    
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    
    # Remove diagonal for unbiased estimator
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    
    term_xx = np.sum(Kxx) / (n * (n - 1))
    term_yy = np.sum(Kyy) / (m * (m - 1))
    term_xy = np.sum(Kxy) / (n * m)
    
    mmd_squared = term_xx + term_yy - 2 * term_xy
    return mmd_squared

    
def frechet_distance(X, Y, eps=1e-6, n_pc_components=10):
    """
    Compute Fréchet distance between two spectroscopic datasets.

    Parameters
    ----------
    X : ndarray (n_samples_x, n_features)
        Spectra dataset 1
    Y : ndarray (n_samples_y, n_features)
        Spectra dataset 2
    eps : float
        Numerical stability constant

    Returns
    -------
    float
        Fréchet distance
    """
    X = X.copy()
    Y = Y.copy()
    
    if n_pc_components is not None:
        pca = PCA(n_components=n_pc_components)
        X = pca.fit_transform(X)
        Y = pca.transform(Y)

    # Mean spectra
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)

    # Covariance matrices
    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    # Mean difference term
    mean_diff = mu_x - mu_y
    mean_term = mean_diff @ mean_diff

    # Product covariance
    cov_prod = cov_x @ cov_y

    # Matrix square root
    covmean = sqrtm(cov_prod)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Fréchet distance
    dist = mean_term + np.trace(cov_x + cov_y - 2 * covmean)

    return float(dist)




import numpy as np
import itertools


def within_condition_icc(data, labels, label_names=("age","sex","bmi"), label_values=(0,1)):
    """
    Compute multivariate ICC and per-feature ICC.

    Parameters
    ----------
    data : ndarray (N, D)
        Data matrix (samples × features)
    labels : dict-like
        Dictionary containing label arrays
    label_names : tuple
        Names of conditioning variables
    label_values : iterable
        Possible values per label (assumed same for all)

    Returns
    -------
    ICC : float
        Multivariate ICC (trace ratio)
    ICC_per_feature : ndarray (D,)
        ICC per feature
    Sigma_w : ndarray (D,D)
        Within-condition covariance
    Sigma_b : ndarray (D,D)
        Between-condition covariance
    """

    N, D = data.shape
    global_mean = data.mean(axis=0)

    Sigma_w = np.zeros((D, D))
    Sigma_b = np.zeros((D, D))

    df_w = 0

    for vals in itertools.product(label_values, repeat=len(label_names)):

        mask = np.ones(N, dtype=bool)

        for name, val in zip(label_names, vals):
            mask &= labels[name] == val

        X = data[mask]
        n = len(X)

        if n == 0:
            continue

        mean_g = X.mean(axis=0)

        # within-condition covariance
        if n > 1:
            cov = np.cov(X, rowvar=False)
            Sigma_w += (n - 1) * cov
            df_w += (n - 1)

        # between-condition covariance contribution
        diff = (mean_g - global_mean).reshape(-1,1)
        Sigma_b += n * (diff @ diff.T)

    Sigma_w /= df_w
    Sigma_b /= (N - 1)

    ICC_per_feature = np.diag(Sigma_b) / (np.diag(Sigma_b) + np.diag(Sigma_w))
    ICC = np.trace(Sigma_b) / (np.trace(Sigma_b) + np.trace(Sigma_w))

    return np.mean(ICC_per_feature)



def within_condition_distribution_distance(
    real_data,
    sim_data,
    labels,
    label_names=("age","sex","bmi"),
    label_values=(0,1),
    distance_measure = 'frechet',
):
    """
    Conditional Fréchet distance between real and simulated data.

    Returns
    -------
    mean_fd : float
        weighted mean conditional Fréchet distance
    fd_per_condition : dict
        FD per condition combination
    """

    fds = []
    weights = []
    fd_dict = {}

    N = len(real_data)

    for vals in itertools.product(label_values, repeat=len(label_names)):

        mask = np.ones(N, dtype=bool)

        for name, val in zip(label_names, vals):
            mask &= labels[name] == val

        X_real = real_data[mask]
        X_sim  = sim_data[mask]

        if len(X_real) < 2 or len(X_sim) < 2:
            continue

        if distance_measure == 'frechet':
            fd = frechet_distance(X_real, X_sim)
        elif distance_measure == 'mmd':
            fd = compute_mmd(X_real, X_sim)
        else:
            raise ValueError('distance_measure must be in ["frechet", "mmd"]')

        weight = len(X_real)

        fds.append(fd)
        weights.append(weight)

        fd_dict[vals] = fd

    weights = np.array(weights)
    fds = np.array(fds)

    mean_fd = np.sum(weights * fds) / np.sum(weights)

    return mean_fd


    


def Logistic_Regression_Classifier(X, y):

    # With inner cv for hyperparameter tuning
    logistic = LogisticRegression(penalty="l2", max_iter=10000)
    pipeline = Pipeline(steps=[('logistic', logistic)])
    p_grid = {"logistic__C": [0.001, 1, 10]}
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring='roc_auc',  n_jobs=-1)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    outer_cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=None)

    for train, test in outer_cv.split(X, y):

        probas_ = clf.fit(X[train], y[train]).decision_function(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    
    return(mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc)


def Logistic_Regression_Classifier_def_train_test(X_train, y_train, X_test, y_test):

    # With nested CV loop for hyperparameter tuning
    logistic = LogisticRegression(penalty="l2", max_iter=10000)
    pipeline = Pipeline(steps=[('logistic', logistic)])
    p_grid = {"logistic__C": [0.001, 1, 10]}
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Splitting simulated data for cross-validation
    outer_cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=None)
    
    test_index_real = []
    for _, test_index in outer_cv.split(X_test, y_test):
        test_index_real.append(test_index)

    
    for i, (train_index, _) in enumerate(outer_cv.split(X_train, y_train)):
        # Train on simulated data (using current CV split)
        X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]

        # Fit the model on training fold
        clf.fit(X_train_fold, y_train_fold)
        
        X_test_split = X_test[test_index_real[i]]
        y_test_split = y_test[test_index_real[i]]    
        
        # Test on real data
        probas_ = clf.decision_function(X_test_split)
        fpr, tpr, thresholds = roc_curve(y_test_split, probas_)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    
    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc


def filter_for_binary_condition(condition_dict):
    c = []
    for key in condition_dict.keys():
        # Filter out None and np.nan from the list
        filtered_values = [value for value in condition_dict[key] if value is not None]# and not np.isnan(value)]
        
        # Check if the set of filtered values has exactly two unique elements
        if len(set(filtered_values)) == 2:
            c.append(key)
    return list(c)

def filter_for_continuous_condition(condition_dict):
    c = []
    for key in condition_dict.keys():
        # Filter out None and np.nan from the list
        filtered_values = [value for value in condition_dict[key] if value is not None]# and not np.isnan(value)]
        
        # Check if the set of filtered values has exactly two unique elements
        if len(set(filtered_values)) > 2:
            c.append(key)
    return list(c)

def matching_list_items(list1, list2):
    return [item for item in list1 if item in list2]


def compute_authenticity(real_data, synthetic_data):
    """
    Compute the authenticity score for synthetic samples.
    
    Parameters:
    - real_data: np.ndarray, shape (n_real_samples, n_features)
        The real dataset.
    - synthetic_data: np.ndarray, shape (n_synthetic_samples, n_features)
        The synthetic dataset.
        
    Returns:
    - authenticity: float
        The authenticity score, i.e., the fraction of synthetic samples that are not memorized copies of real data.
    """

    # Fit nearest neighbors for real and synthetic data
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real_data)
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(synthetic_data)
    
    # Compute distances for real-to-real and real-to-synthetic
    real_to_real_distances, _ = nbrs_real.kneighbors(real_data)
    real_to_synth_distances, real_to_synth_indices = nbrs_synth.kneighbors(real_data)
    
    # Exclude the closest real point itself by taking the second neighbor for real-to-real distances
    real_to_real_distances = torch.from_numpy(real_to_real_distances[:, 1].squeeze())
    real_to_synth_distances = torch.from_numpy(real_to_synth_distances.squeeze())
    real_to_synth_indices = real_to_synth_indices.squeeze()
    
    # Authenticity condition: real-to-real distance < real-to-synthetic distance
    authen_mask = real_to_real_distances[real_to_synth_indices] < real_to_synth_distances
    authenticity = torch.mean(authen_mask.float()).item()
    
    return authenticity


def group_tuples_by_first_element(tuples):
    tuples.sort(key=lambda x: x[0])
    grouped = [list(group) for _, group in groupby(tuples, key=lambda x: x[0])]
    return grouped


def group_tuples_by_last_element(tuples):
    tuples.sort(key=lambda x: x[-1])
    grouped = [list(group) for _, group in groupby(tuples, key=lambda x: x[-1])]
    return grouped


# Define identity mapping if no scaler is specified
class _NoneScaler():
    def __init__(self):
        pass
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X):
        return X
    def inverse_transform(self, X):
        return X