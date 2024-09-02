import numpy as np
import os
import copy

from cluster.selfrepresentation import ElasticNetSubspaceClustering
from cluster.selfrepresentation import SparseSubspaceClusteringOMP
from sklearn.metrics.cluster import _supervised
from scipy.optimize import linear_sum_assignment

import pickle
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


def clustering_accuracy(labels_true, labels_pred):
    """Clustering Accuracy between two clusterings.
    Clustering Accuracy is a measure of the similarity between two labels of
    the same data. Assume that both labels_true and labels_pred contain n 
    distinct labels. Clustering Accuracy is the maximum accuracy over all
    possible permutations of the labels, i.e.
    \max_{\sigma} \sum_i labels_true[i] == \sigma(labels_pred[i])
    where \sigma is a mapping from the set of unique labels of labels_pred to
    the set of unique labels of labels_true. Clustering accuracy is one if 
    and only if there is a permutation of the labels such that there is an
    exact match
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    labels_pred : array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    
    Returns
    -------
    accuracy : float
       return clustering accuracy in the range of [0, 1]
    """
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    # value = _supervised.contingency_matrix(labels_true, labels_pred, sparse=False)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)




def make_parameters(param_values):
    if len(param_values) == 1:
        for key in param_values:
            values = param_values[key]
            res_list = []
            for val in values:
                res_list.append({key: val})
            return res_list

    for key in param_values:
        values = param_values[key]
        new_param_values = copy.deepcopy(param_values)
        del new_param_values[key]
        res_list = make_parameters(new_param_values)
        new_res_list = []
        for dic in res_list:
            for val in values:
                dic[key] = val
                cdic = copy.deepcopy(dic)
                new_res_list.append(cdic)
        return new_res_list


def grid_search(X, labels, num_clusters, hyperparam_dict, save_path):
    if os.path.isfile(save_path):
        hyperparam_list = load_var(save_path)
    else:
        hyperparam_list = make_parameters(hyperparam_dict)

    best_nmi = 0
    best_row_idx = -1
    best_row = ""
    for hyper_idx, hparam in enumerate(hyperparam_list):
        thr = hparam['thr']
        n_nonzero = hparam["n_nonzero"]
        
        if "NMI" not in hparam:
            model = SparseSubspaceClusteringOMP(n_clusters=num_clusters, 
                                                n_nonzero=n_nonzero, thr=thr).fit(X)

            # print(model.labels_)
            label_list = model.labels_
            
            nmi_score = normalized_mutual_info_score(labels, label_list)
            acc = clustering_accuracy(labels_true=labels, labels_pred=model.labels_)
            ari = adjusted_rand_score(labels_true=labels, labels_pred=model.labels_)
            hparam["NMI"] = nmi_score
            hparam["ACC"] = acc
            hparam["ARI"] = ari
            
            save_var(save_path, hyperparam_list)
            
        row_res = "{}/{}   thr: {}, n_nonzero: {:2d}, ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}"
        row_res = row_res.format(hyper_idx, len(hyperparam_list), thr, n_nonzero, 
                                 hparam["ACC"], hparam["NMI"], hparam["ARI"])
        print(row_res)
        
        if hparam["NMI"] > best_nmi:
            best_nmi = hparam["NMI"]
            best_row_idx = hyper_idx
            best_row = row_res

    print("\n\n\nbest res is: \n{}\n\n\n".format(best_row))
    return hyperparam_list



if __name__ == "__main__":
    dataset = "stl10_test.pckl"

    imgs, labels, X = load_var("../data/data/stl10/{}".format(dataset))
    num_clusters = len(np.unique(labels))
    
    hyperparam_dict = {"n_nonzero": [3, 5, 10, 15],
                      "thr": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],}
    
    hyperparam_list = make_parameters(hyperparam_dict)
    res = grid_search(X=X, labels=labels, 
                      num_clusters=num_clusters,
                      hyperparam_dict=hyperparam_dict, 
                      save_path="output/SSC-OMP_{}".format(dataset))
    
    print("the end")



# model = ElasticNetSubspaceClustering(n_clusters=num_clusters,algorithm='lasso_lars',gamma=50).fit(X.T)

# model = ElasticNetSubspaceClustering(n_clusters=num_clusters,algorithm='lasso_lars',gamma=50).fit(X)