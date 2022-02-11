#%%
from traceback import print_tb
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
import numpy as np
import numpy.linalg as ln
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Tunnel_parser import Tunnel_Parser,Tunnel_Parser_full
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
    #Indexes to extract abs and ord coord separatlyn due to plt format
    abs_index=[0,2]
    ord_index=[1,3]

    #Iterates through each cluster, associatinng a color to them
    for k, col in zip(unique_labels, colors):
        #Some algos may return -1 as cluster ID for noise data
        if k == -1:
            ind = [index for index, element in enumerate(labels) if element == k]

            for j in ind:
                abs=list(X[j][abs_index])
                ord=list(X[j][ord_index])
                plt.plot(abs,ord,color='black')
        else :
            #Gathers indexes for each point belonging to the current cluster
            ind = [index for index, element in enumerate(labels) if element == k]
            for j in ind:
                abs=list(X[j][abs_index])
                ord=list(X[j][ord_index])
                plt.plot(abs,ord,color=colors[k])
    plt.title("Estimated number of clusters : {}".format(n_clusters_))
    plt.show()

def Cluster_Rectangles(X,unique_labels,labels,switch=False):
    """Cluster_Rectangles [Goes through clusters and creates a rectangle surronding it]

    Arguments:
        X {[array]} -- Starting and ending point of each tunnel [Xa Ya Xb Yb]
        unique_labels {[array]} -- [Unique Id of each tunnel cluster]
        labels {[array]} -- [Cluster ID for each tunnel, in order]

    Keyword Arguments:
        switch {[bool]} -- [Coordinates are switched when displaying rectangles on ipyleaflet map] (default: {False})

    Returns:
        [array] -- [Bottom_left and Top_right point of each cluster's rectangle [Xa Ya]]
    """
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
    X=np.array(Tunnel_Parser_full())

    #Applying basic algo to find clusters, needs tuning
    clustering = AgglomerativeClustering(n_clusters=45).fit(X)
    labels=clustering.labels_
    labels=labels.tolist()
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)

    #Creating arrays with random color codes for each cluster 
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    rec_colors = ["#%06x" % random.randint(0, 0xFFFFFF) for i in range(len(unique_labels))]
    #Show clusters on graph to check for errors
    Graph_Clusters(X,unique_labels,colors,labels,n_clusters_)
#%%
    #Compute coordinate of rectangle to display 
    rectangles_coordinates=Cluster_Rectangles(X,unique_labels,labels,switch=True)

    #display map with tunnels traced and boxed in rectangles corresponding to their cluster
    #not useful to final product but useful for debugging and performance checking
    map_layer = basemap_to_tiles(basemaps.CartoDB.Positron)
    m = Map(layers=(map_layer, ), center=((48.852,2.246)), zoom=5, scroll_wheel_zoom=True)
    #Adds rectangle for each cluster with responsible 
    for coord,col in zip(rectangles_coordinates,rec_colors):
        rectangle = Rectangle(bounds=coord,weight=2,color=col)
        m.add_layer(rectangle)
    
    #Adds tunnel, drawn in black for visibility
    geojson= GeoJSON(
    data=Tunnel_Parser(raw=True),
    style={ 
        'color':'black', 'opacity': 1, 'fillOpacity': 0.1, 'weight': 2
    },
    hover_style={
        'color':'red', 'opacity': 1, 'fillOpacity': 0.1, 'weight': 3
    })
    m.add_layer(geojson)

    #Saves map as an html file
    m.save('my_map.html', title='My Map')
    m
# %%
