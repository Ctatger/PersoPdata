import pandas as pd
from ast import literal_eval

from sklearn.cluster import DBSCAN

from travel_clustering import create_clusters, Cluster_Labels,\
                                Remove_redundant_travels


def create_dataframe():
    # Parsing
    df_travel = pd.read_csv("csv/travel_based_dataframe.csv", index_col=0,
                            converters={"start_gps_coord": literal_eval,
                                        "end_gps_coord": literal_eval,
                                        "travel_gps_list": literal_eval})

    df_travel = df_travel[df_travel['travel_distance_km'] > 0.5]

    # Cluster classification using DBSCAN
    dbscan = DBSCAN(eps=0.005, min_samples=3)
    create_clusters(df_travel, 'start_gps_coord', dbscan,
                    header_name='gps_start_cluster')

    dbscan = DBSCAN(eps=0.005, min_samples=3)
    create_clusters(df_travel, 'end_gps_coord', dbscan,
                    header_name='gps_end_cluster')

    df_travel = df_travel[df_travel['gps_end_cluster'] != -1]
    df_travel = df_travel[df_travel['gps_start_cluster'] != -1]

    # Adds location label, using gps street data
    Cluster_Labels(df_travel)
    df_travel = Remove_redundant_travels(df_travel)

    return df_travel
