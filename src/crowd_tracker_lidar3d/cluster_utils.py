import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import random
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from IPython.display import clear_output, Image, display
from sklearn.datasets.samples_generator import make_blobs
import itertools
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.cluster import adjusted_rand_score
import os
from scipy import sparse, io
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def return_cluster_and_noise_points(cluster_model): 
    labels = cluster_model.labels_
    #  Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return n_clusters_, n_noise_


def plot_3d_cluster_results(cluster_data, cluster_model):   
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')
    labels = cluster_model.labels_
    unique_labels = set(labels)
    colors = [plt.cm.hsv(each)  for each in np.linspace(0, 1, len(unique_labels))]
#     colors = [plt.cm.Spectral(each)  for each in np.linspace(0, 1, len(unique_labels))]

    core_samples_mask = None 
    try:
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[cluster_model.core_sample_indices_] = True
    except AttributeError: 
        core_samples_mask = None 
    for k, col in zip(unique_labels, colors):  
        if k == -1:    # Black used for noise.    
            col = [0, 0, 0, 1]  
        class_member_mask = (labels == k) 
        
        if core_samples_mask is not None: 
            xyz = cluster_data[class_member_mask & core_samples_mask]  
            xyz_outlier = cluster_data[class_member_mask & ~core_samples_mask]
            ax.scatter(xyz_outlier[:, 0], xyz_outlier[:, 1], xyz_outlier[:, 2], c=col, marker="x")
        else: 
            xyz = cluster_data[class_member_mask]
        
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker=".", label=k)

    ax.set_ylabel('Y', fontsize=18)
    ax.set_xlabel('X', fontsize=18)
    ax.set_zlabel('Z', fontsize=18)
    n_clusters_, _ = return_cluster_and_noise_points(cluster_model)

    ax.legend(fontsize='x-large', markerscale=5, loc='center left', bbox_to_anchor=(1, 0.5), title="Labels")
    plt.title('Estimated number of clusters: %d' % n_clusters_, fontsize=20)
    ax.tick_params(labelsize=14)
    plt.show()


def plot_xy_cluster_results(cluster_data, cluster_model):
    labels = cluster_model.labels_
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    core_samples_mask = None 
    try:
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[cluster_model.core_sample_indices_] = True
    except AttributeError: # not every clustering model has core samples
        core_samples_mask = None 
    
    fig = plt.figure(figsize=[13, 8])

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        if core_samples_mask is not None: 
            xy = cluster_data[class_member_mask & core_samples_mask]
            xy_outlier = cluster_data[class_member_mask & ~core_samples_mask]
            plt.plot(xy_outlier[:, 0], xy_outlier[:, 1], 'x', markerfacecolor=tuple(col),
                     markeredgecolor=tuple(col), markersize=1)
        else: 
            xy = cluster_data[class_member_mask]

        plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),
                 markeredgecolor=tuple(col), markersize=8)

        
    n_clusters_, _ = return_cluster_and_noise_points(cluster_model)
    plt.ylim((-1,1))
    plt.xlim((-2,3))
    plt.title('Estimated number of clusters: %d' % n_clusters_, fontsize=18)
    plt.tick_params(labelsize=14)
    plt.ylabel('Y', fontsize=16)
    plt.xlabel('X', fontsize=16)
    plt.show()
    return fig


def externalValidation(truthClusters, predictedClusters):
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    scores = {}
    scores['_rand_index'] = adjusted_rand_score(truthClusters, predictedClusters)
    scores['_homogeneity_score'] = metrics.homogeneity_score(truthClusters, predictedClusters)
    scores['_purity_score'] = purity_score(truthClusters, predictedClusters)
    scores['_adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(truthClusters, predictedClusters)
    scores['_fowlkes_mallows_score'] = metrics.fowlkes_mallows_score(truthClusters, predictedClusters)  
    return scores


def internalValidation(data, clusters):
    scores = {}
    """
    The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    Scores around zero indicate overlapping clusters.
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    """
    scores['_silhouette_score'] =metrics.silhouette_score(data,clusters ,metric='euclidean')
    """
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    The score is fast to compute
    """
    scores['_calinski_harabaz_score'] = metrics.calinski_harabaz_score(data,clusters)
    """
    Zero is the lowest possible score. Values closer to zero indicate a better partition.
    The Davies-Boulding index is generally higher for convex clusters than other concepts of clusters, 
    such as density based clusters like those obtained from DBSCAN.
    """
    scores['_davies_bouldin_score'] = metrics.davies_bouldin_score(data,clusters)
    return scores


