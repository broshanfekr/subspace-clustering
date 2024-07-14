import numpy as np

from cluster.selfrepresentation import ElasticNetSubspaceClustering
from cluster.selfrepresentation import SparseSubspaceClusteringOMP

import pickle
from sklearn.metrics.cluster import normalized_mutual_info_score


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


# # generate 7 data points from 3 independent subspaces as columns of data matrix X
# X = np.array([[1.0, -1.0, 0.0, 0.0, 0.0,  0.0, 0.0],
#               [1.0,  0.5, 0.0, 0.0, 0.0,  0.0, 0.0],
#               [0.0,  0.0, 1.0, 0.2, 0.0,  0.0, 0.0],
#               [0.0,  0.0, 0.2, 1.0, 0.0,  0.0, 0.0],
#               [0.0,  0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
#               [0.0,  0.0, 0.0, 0.0, 1.0,  1.0, -1.0]])

imgs, labels, X = load_var("data/mnist/extracted_features.pckl")
num_clusters = len(np.unique(labels))
# model = ElasticNetSubspaceClustering(n_clusters=num_clusters,algorithm='lasso_lars',gamma=50).fit(X.T)

# model = ElasticNetSubspaceClustering(n_clusters=num_clusters,algorithm='lasso_lars',gamma=50).fit(X)
model = SparseSubspaceClusteringOMP(n_clusters=num_clusters).fit(X)

# print(model.labels_)
label_list = model.labels_

nmi_score = normalized_mutual_info_score(labels, label_list)
print("Clustering NMI is: {}".format(nmi_score))

# this should give you array([1, 1, 0, 0, 2, 2, 2]) or a permutation of these labelspip install progressbar2