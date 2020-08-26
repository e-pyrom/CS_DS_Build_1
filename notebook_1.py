from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN



#DBSCAN data clustering algorithm
#scipy.spatial.disctance submodule to conveniently handle distance calculations

class Basic_DBSCAN:
    """
    Parameters:

    eps: Radius of neighborhood graph

    minPts: Number of neighbors required to label a given point as a core point.

    metric: Distance metric used to determine distance points;
        currently accepts scipy.spatial.distance metrics for two
        numeric vectors

    """

    def __init__(self, eps, minPts, metric=distance.euclidean):
        self.eps = eps
        self.minPts = minPts
        self.metric = metric

    def fit_predict(self, X):
        """
        Parameters:

        X: An n-dimensional array of numeric vectors to be analyzed

        Returns:

        [n] cluster labels
        """

        clusters = [0] * X.shape[0]

        simple_DBSCAN(X, clusters, self.eps, self.minPts, self.metric)

        return clusters