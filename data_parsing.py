# %%
import pandas as pd
# import numpy as np
import os
import glob
import json

from ast import literal_eval
import math
from datetime import datetime, timedelta
from ipyleaflet import Map, basemaps, basemap_to_tiles, CircleMarker

from sklearn.cluster import DBSCAN

from travel_clustering import Cluster_Labels,\
    Remove_redundant_travels


def format_time(time_):
    (mins, hour) = math.modf(time_)

    formatted = '{:02d}:{:02d}:00'.format(round(hour), round(mins*60))
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
                    header_name='Start_cluster')

    dbscan = DBSCAN(eps=0.005, min_samples=3)
    create_clusters(df_travel, 'end_gps_coord', dbscan,
                    header_name='End_cluster')

    df_travel = df_travel[df_travel['End_cluster'] != -1]
    df_travel = df_travel[df_travel['Start_cluster'] != -1]

    # Adds location label, using gps street data
    Cluster_Labels(df_travel)
    Cluster_Labels(df_travel, group='End_cluster')
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


def create_window_dataframe(df, verbose=True):

    MAX_DELTA = timedelta(minutes=1)

    df_wind = pd.DataFrame(columns=['Coordinates', 'Wd_change', 'Time',
                           'Time_delta', 'Day', 'Window_cluster'])
    method = DBSCAN(eps=0.005, min_samples=int(len(df)/5))
    # method = OPTICS(min_samples=int(len(df)/8))
    if type(df['Coordinates'][0]) == str:
        df['Coordinates'] = df['Coordinates'].apply(literal_eval)
    create_clusters(df, 'Coordinates', method)
    window_cluster = []
    for i in range(len(df)-1):

        if df.at[i, 'Wd_state'] != df.at[i+1, 'Wd_state']:
            FMT = '%H:%M:%S'

            if df.at[i+1, 'Wd_state'] == 0:
                delta = pd.NaT
                data = {'Coordinates': [df.at[i+1, 'Coordinates']], 'Wd_change': 'Opened',
                        'Time': df.at[i+1, 'Time'],
                        'Day': df.at[i+1, 'Day'], 'Time_delta': delta,
                        'Coord_cluster': df.at[i+1, 'Coordinates_cluster']*2}

                dummy = pd.DataFrame(data=data, index=[i])
                df_wind = pd.concat([df_wind, dummy])

            elif (df.at[i+1, 'Wd_state'] == 1 and i > 0):

                if (datetime.strptime(df.at[i, 'Time'], FMT) > datetime.strptime(df.at[i-1, 'Time'], FMT)):
                    delta = datetime.strptime(df.at[i, 'Time'], FMT) - datetime.strptime(df.at[i-1, 'Time'], FMT)
                    if delta > MAX_DELTA:
                        delta = MAX_DELTA
                else:
                    delta = pd.NaT

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
    df_wind = df_wind.astype({'Window_cluster': 'int32', 'Coord_cluster': 'int32'})

    nb_c = len(df_wind.loc[df_wind['Coord_cluster'] >= 0].Coord_cluster.unique())

    noise = len(df_wind.loc[df_wind['Coord_cluster'] < 0])
    if verbose:
        print("The model found {} position clusters, it will use {} window state clusters".format(nb_c, nb_c*2))
        print("{} total points were categorized as noise".format(noise))
        print("Representing {:.2f}% of total data".format(100*noise/len(df_wind)))

    df_wind['Start_cluster'] = pd.NaT
    df_wind['End_cluster'] = pd.NaT

    for ind, row in df_wind.iterrows():

        if (row.Window_cluster % 2) == 0:
            df_wind.at[ind, 'Start_cluster'] = row.Window_cluster + 1
            df_wind.at[ind, 'End_cluster'] = row.Window_cluster
        else:
            df_wind.at[ind, 'Start_cluster'] = row.Window_cluster - 1
            df_wind.at[ind, 'End_cluster'] = row.Window_cluster

    return df_wind


def parse_app_data(path):
    data_array = []

    with open(path) as f:
        data = [json.loads(line) for line in f]

    startTime = data[0]['startRecordingTime']
    startDate = startTime[0:10].split("-")
    startWeekday = datetime(int(startDate[0]), int(startDate[1]), int(startDate[2])).weekday()
    startHour = ':'.join(startTime[-11:].split(':', 2)[:3])
    startHour = startHour.split('.')[0]
    startRow = {'Coordinates': [[0.0, 0.0]], 'Wd_state': 1,
                'Day': startWeekday, 'Time': startHour}

    for i in range(3, len(data)):
        rid = data[i]['rid']
        value = data[i]['extras']
        time = data[i]['date']
        date = time[0:10].split("-")
        weekday = datetime(int(date[0]), int(date[1]), int(date[2])).weekday()
        hour = ':'.join(time[-17:].split(':', 2)[:3])
        hour = hour.split('.')[0]

        rid = rid.strip('#0')
        data_array.append({'rid': rid, 'value': value, 'date': weekday, 'hour': hour})

    dataframe = pd.DataFrame()
    dataframe = pd.concat([dataframe, pd.DataFrame(startRow)], ignore_index=True)

    for id, entry in enumerate(data_array):
        if entry['rid'] == "AndroidCar.WindowPosition":
            for k in range(id, id-20, -1):
                if data_array[k]['rid'] == "Location.CurrentLocation":
                    if entry['value'] == 100:
                        coordinates = data_array[k]['value']

                        dataframe_row = {'Coordinates': [coordinates], 'Wd_state': 1,
                                         'Day': entry['date'], 'Time': entry['hour']}
                        dataframe = pd.concat([dataframe, pd.DataFrame(dataframe_row)], ignore_index=True)
                        break
                    else:
                        coordinates = data_array[k]['value']
                        dataframe_row = {'Coordinates': [coordinates], 'Wd_state': 0,
                                         'Day': entry['date'], 'Time': entry['hour']}
                        dataframe = pd.concat([dataframe, pd.DataFrame(dataframe_row)], ignore_index=True)
                        break
    return dataframe


def create_cluster_map(df, display_noise=False):
    rec_colors = ['magenta', 'cyan', 'blue', 'pink', 'purple', 'red', 'orange', 'yellow', 'brown', 'green']
    map_layer = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    position = df.at[0, 'Coordinates']
    m = Map(layers=(map_layer, ), center=((position[0], position[1])), zoom=5, scroll_wheel_zoom=True)

    for _, row in df.iterrows():
        if row['Coord_cluster'] >= 0:
            m.add_layer(CircleMarker(location=row['Coordinates'], radius=3,
                                     color=rec_colors[row['Coord_cluster'] % len(rec_colors)],
                                     fill_color='#FFFFFF', weight=2))
        elif (row['Coord_cluster'] < 0 and display_noise):
            m.add_layer(CircleMarker(location=row['Coordinates'], radius=3,
                                     color='black',
                                     fill_color='#FFFFFF', weight=2))
    return m
# %%
