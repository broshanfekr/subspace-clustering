import pickle
import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from metrics.cluster.accuracy import clustering_accuracy
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
 

def apply_kmeans_clustering(X, labels_true, num_clusters, n_init):
    cluster_model = KMeans(n_clusters=num_clusters, n_init=n_init)
    acc_lst = []
    nmi_lst = []
    ari_lst = []
    for _ in range(n_init):
        cluster_model.fit(X)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(labels_true, pred_label)
        nmi_score = normalized_mutual_info_score(labels_true, pred_label)
        ari_score = adjusted_rand_score(labels_true, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        ari_lst.append(ari_score)
        
    return {"NMI": np.mean(nmi_lst), "ARI": np.mean(ari_lst), "ACC": np.mean(acc_lst)}


def apply_spectral_clustering(X, labels_true, num_clusters):
    sc = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize').fit(X)
    pred_label = sc.labels_   
    acc = clustering_accuracy(labels_true, pred_label)
    nmi = normalized_mutual_info_score(labels_true, pred_label)
    ari = adjusted_rand_score(labels_true, pred_label)
    return {"NMI": np.mean(nmi), "ARI": np.mean(ari), "ACC": np.mean(acc)}


def gen_text(kwargs):
    res_text = []
    for key, value in kwargs.items():
        res_text.append("{}: {:.4f}".format(key, value.item()))

    res_text = ", ".join(res_text)
    return res_text


if __name__ == "__main__":
    imgs, labels, X = load_var("../../data/cifar10/cifar10_5000samples.pckl")
    num_clusters = len(np.unique(labels))
    n_init = 10
    
    kmeans_res = apply_kmeans_clustering(X, labels, num_clusters, n_init)
    print("kmeans res: {}".format(gen_text(kmeans_res)))
    
    spectral_res = apply_spectral_clustering(X, labels, num_clusters)
    print("spectral res: {}".format(gen_text(spectral_res)))
    
