#%%
from traceback import print_tb
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
import numpy as np
import numpy.linalg as ln
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Tunnel_parser import Tunnel_Parser
from ipyleaflet import Map, basemaps, basemap_to_tiles, Rectangle, GeoJSON
from math import trunc
import random



def Graph_Clusters(X,unique_labels,colors,labels,n_clusters_):
    """Graph_Clusters [Traces graph of tunnel clusters, changing color for each]

    Arguments:
        X {array} -- Starting and ending point of each tunnel [Xa Ya Xb Yb]
        unique_labels {array} -- [Unique Id of each tunnel cluster]
        colors {[array]} -- [RGB codes, same len as unique_labels]
        labels {[array]} -- [Cluster ID for each tunnel, in order]
        n_clusters_ {[array]} -- [Number of unique labels, ecxept -1]
    """
    abs_index=[0,2]
    ord_index=[1,3]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            ind = [index for index, element in enumerate(labels) if element == k]
            for j in ind:
                abs=list(X[j][abs_index])
                ord=list(X[j][ord_index])
                plt.plot(abs,ord,color='black')
        else :
            ind = [index for index, element in enumerate(labels) if element == k]
            for j in ind:
                abs=list(X[j][abs_index])
                ord=list(X[j][ord_index])
                plt.plot(abs,ord,color=colors[k])
    plt.title("Estimated number of clusters : {}".format(n_clusters_))
    plt.show()

def Cluster_Rectangles(X,unique_labels,labels,switch=False):
    abs_index=[0,2]
    ord_index=[1,3]
    cluster_rectangles=[]
    
    for k in unique_labels:
        abs_list=[]
        ord_list=[]
        bottom_left=[]
        top_right=[]

        if k != -1:
            Cluster_points=[]
            ind = [index for index, element in enumerate(labels) if element == k]

            for j in ind:
                Cluster_points.append(X[j])

        for points in Cluster_points:
            abs_list.append(points[0])
            abs_list.append(points[2])
            ord_list.append(points[1])
            ord_list.append(points[3])

        bottom_left.append(min(abs_list))
        bottom_left.append(min(ord_list))
        top_right.append(max(abs_list))
        top_right.append(max(ord_list))

        if switch:
            cluster_rectangles.append((tuple(bottom_left[::-1]),tuple(top_right[::-1])))
        else:
            cluster_rectangles.append((tuple(bottom_left),tuple(top_right)))
    return cluster_rectangles
#%%
if __name__=="__main__":
    sns.set_theme()
    #Parse GeoJson file to extract tunnels coordinates
    X=np.array(Tunnel_Parser())
    #Applying basic DBSCAN algo to find clusters, needs tuning
    #clustering = DBSCAN(eps=0.035, min_samples=2).fit(X)
    #clustering = AgglomerativeClustering(n_clusters=20).fit(X)
    clustering=Birch(n_clusters=25).fit(X)
    labels=clustering.labels_
    labels=labels.tolist()
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    rec_colors = ["#%06x" % random.randint(0, 0xFFFFFF) for i in range(len(unique_labels))]
    print(colors)
    print(rec_colors)
    #Show clusters on graph to check for errors
    Graph_Clusters(X,unique_labels,colors,labels,n_clusters_)
# %%
