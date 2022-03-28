# %%
import pandas as pd
# import numpy as np
import os
import glob
from ast import literal_eval
import math
from datetime import datetime

from sklearn.cluster import DBSCAN

from travel_clustering import Cluster_Labels,\
    Remove_redundant_travels


def format_time(time):
    (mins, hour) = math.modf(time)
    formatted = '{:02d}:{:02d}'.format(round(hour), round(mins*60))
    return formatted


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


def parse_csv(dir_path, index_column=None):
    root_dir = os.path.abspath(os.curdir)
    os.chdir(dir_path)
    file_list = glob.glob('*.csv')
    print('Nb of files: ', len(file_list))

    data_list = []
    for file_name in file_list:
        df_temp = pd.read_csv(file_name, index_col=index_column)
        data_list.append(df_temp)

    dataframe = pd.concat(data_list, ignore_index=True)
    os.chdir(root_dir)
    return dataframe


def create_clusters(df, columns, clustering_method, header_name=None):
    if not header_name:
        header_name = "{}_cluster".format(columns)
        data_array = [positions for positions in df[columns].values]
    df[header_name] = clustering_method.fit_predict(data_array)

    return df


def create_window_dataframe(df):
    df_wind = pd.DataFrame(columns=['Coordinates', 'Wd_change', 'Time',
                           'Time_delta', 'Day', 'Window_cluster'])
    Coord = []
    for k in range(len(df)):
        Coord.append([df.at[k, 'Pos_lat'], df.at[k, 'Pos_lon']])
    df['Coordinates'] = Coord
    dbscan = DBSCAN(eps=0.005, min_samples=3)
    create_clusters(df, 'Coordinates', dbscan)
    window_cluster = []
    for i in range(len(df)-1):

        if df.at[i, 'Wd_state'] != df.at[i+1, 'Wd_state']:
            FMT = '%H:%M'
            delta = datetime.strptime(
                df.at[i+1, 'Time'], FMT) - datetime.strptime(df.at[i, 'Time'], FMT)

            if df.at[i+1, 'Wd_state'] == 0:
                data = {'Coordinates': [df.at[i+1, 'Coordinates']], 'Wd_change': 'Opened',
                        'Time': df.at[i+1, 'Time'],
                        'Day': df.at[i+1, 'Day'], 'Time_delta': str(delta),
                        'Coord_cluster': df.at[i+1, 'Coordinates_cluster']*2}

                dummy = pd.DataFrame(data=data, index=[i])
                df_wind = pd.concat([df_wind, dummy])

            elif df.at[i+1, 'Wd_state'] == 1:
                data = {'Coordinates': [df.at[i+1, 'Coordinates']], 'Wd_change': 'Closed',
                        'Time': df.at[i+1, 'Time'], 'Day': df.at[i+1, 'Day'], 'Time_delta': str(delta),
                        'Coord_cluster': df.at[i+1, 'Coordinates_cluster']*2}

                dummy = pd.DataFrame(data=data, index=[i])
                df_wind = pd.concat([df_wind, dummy])
            else:
                pass

    df_wind = df_wind.reset_index(drop=True)

    for i in range(len(df_wind)):
        if df_wind.at[i, 'Wd_change'] == 'Opened':
            window_cluster.append(df_wind.at[i, 'Coord_cluster'])
        else:
            window_cluster.append(df_wind.at[i, 'Coord_cluster'] + 1)
    df_wind['Window_cluster'] = window_cluster

    return df_wind
# %%
