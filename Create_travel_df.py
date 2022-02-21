import os
import sys
import glob
from this import d
from time import sleep
#Import Data 
#from log_processor.log_processor import get_reduced_dataset
import datetime

#Data Management
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

#Data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.style.use('seaborn')

# Machine Learning
import tensorflow as tf
from tensorflow import keras

# Performing clustering
from sklearn.cluster import AgglomerativeClustering

from geopy.geocoders import Nominatim

from ast import literal_eval

# The following code will get all cardata files
# We will then filter
columns = [' gpsts', ' lat', ' lon']
logs = []
def parse_cardata(fname, columns):
    car_data = pd.read_csv(fname)
    index = car_data[' systs'].apply(lambda x: int(x))
    car_data = car_data.set_index(index)
    car_data_filtered = car_data[columns].copy()
    if ' ID_ExternalTempDisplayValue' in columns:
        car_data_filtered[' ID_ExternalTempDisplayValue'] = car_data_filtered[' ID_ExternalTempDisplayValue'] - 40
    return car_data_filtered


def haversine_formula(lat1, lon1, lat2, lon2, R=6378.137):
    """
    Calculates the distance in meters between two point on the globe
    """
    dLat = (lat2-lat1) * np.pi / 180
    dLon = (lon2-lon1) * np.pi / 180
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.sin(dLon/2)*np.sin(dLon/2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c*1000 # meters

def excessive_speed(coord1, coord2, threshold_kmh):
    """
    Determines if the speed between the two coordinate points is higher than threshold_kmh value
    """
    t1, lat1, lon1 = coord1
    if t1 == 0:
        return False
    
    t2, lat2, lon2 = coord2
    d = haversine_formula(lat1, lon1, lat2, lon2)
    v = 3600*d/np.abs(t1 - t2) #km/h [3600*m/ms -> km/h]
    return v>threshold_kmh

def remove_incoherent_gps_entry(df, high_delta_time_tolerance=5000, low_delta_time_tolerance=0, threshold_kmh=250):
    """
    Check every entry and keeps only those which satisfies constraints (delta timestamp and excessive speed)
    Columns of the df should be : ['gpsts', 'lat', 'lon']
    """
    #Removing weird gps times avec la vitesse maximale possible
    weird_gps_times = []
    prev_loc = (0, 0.0, 0.0)
    df_val = df.values
    for index, i in enumerate(df.index):
        gps_time = df_val[index][0] 
        current_loc = df_val[index][1:]
        if i<(gps_time - low_delta_time_tolerance) \
        or i>(gps_time+high_delta_time_tolerance) \
        or excessive_speed(prev_loc, (i, current_loc[0], current_loc[1]), threshold_kmh):
            weird_gps_times.append(i)
        else:
            prev_loc = (i, current_loc[0], current_loc[1])
    df = df.drop(weird_gps_times)
    return df

def encode_periodic_value(df, column, max_value, header_name=None):
    if not header_name:
        header_name=column
    df['{}_sin'.format(header_name)] = np.sin(2 * np.pi * df[column]/max_value)
    df['{}_cos'.format(header_name)] = np.cos(2 * np.pi * df[column]/max_value)
    return df

def create_travel_based_df(input_df, interrupt_trigger = 600000, gps_list_point_min_interval=5000):
    """
    Builds a new pd.Dataframe based on travels informations that we can generate from logs
    The input Dataframe has columns [' gpsts', ' lat', ' lon']
    The output Dataframe has columns :
    [
    'start_hour_hmin',
    'end_hour_hmin',
    'duration_minsec',
    'weekday_str',
    'start_gps_label',
    'end_gps_label',
    'travel_distance_km',
    'start_ts',
    'end_ts',
    'duration',
    'start_hour',
    'start_hour_sin',
    'start_hour_cos',
    'end_hour',
    'end_hour_sin',
    'end_hour_cos',
    'weekday',
    'start_gps_coord',
    'end_gps_coord',
    'travel_gps_list'
    ]
    """
    
    """

    'start_gps_label',
    'end_gps_label',
    'travel_distance_km',
    'start_gps_coord',
    'end_gps_coord',
    'travel_gps_list'
    """
    t_prec = 0
    start_times = []
    end_times = []
    for t in input_df.index:
        if t>t_prec+interrupt_trigger:
            start_times.append(t)
            end_times.append(t_prec)
        t_prec = t
    end_times.pop(0)
    end_times.append(input_df.index[-1])
    
    deb_date = datetime.datetime.fromtimestamp(input_df.index[0]/1000)
    end_date = datetime.datetime.fromtimestamp(input_df.index[-1]/1000)
    print("Begin date:", deb_date, ", End date:", end_date, ", timespan:", end_date - deb_date)
    print("number of travel detected : ", len(start_times))

    output_df = pd.DataFrame(data=np.array(start_times), columns=['start_ts'])
    output_df['end_ts'] = np.array(end_times)
    output_df['start_hour'] = output_df['start_ts'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).hour+datetime.datetime.fromtimestamp(x/1000).minute/60)
    output_df['end_hour'] = output_df['end_ts'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).hour+datetime.datetime.fromtimestamp(x/1000).minute/60)
    output_df['start_hour_hmin'] = output_df['start_ts'].apply(lambda x: '{}:{:02d}'.format(datetime.datetime.fromtimestamp(x/1000).hour,datetime.datetime.fromtimestamp(x/1000).minute))
    output_df['end_hour_hmin'] = output_df['end_ts'].apply(lambda x: '{}:{:02d}'.format(datetime.datetime.fromtimestamp(x/1000).hour,datetime.datetime.fromtimestamp(x/1000).minute))
    output_df = encode_periodic_value(output_df, 'start_hour', 24)
    output_df = encode_periodic_value(output_df, 'end_hour', 24)
    output_df['duration'] = np.array(end_times)-np.array(start_times)
    output_df['duration_minsec'] = output_df['duration'].apply(lambda x: '{}m {}s'.format(int(x/60000),int(x/1000)%60))
    output_df['weekday'] = output_df['start_ts'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).weekday())
    output_df['weekday_str'] = output_df['start_ts'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime("%A"))

    #Location
    geolocator = Nominatim(user_agent="notebook")
    # Need to get 'start_gps_label','end_gps_label','travel_distance_km','start_gps_coord','end_gps_coord','travel_gps_list'
    gps_data = []
    i=1
    for start_ts, end_ts in zip(start_times, end_times):
        start_gps = input_df.loc[start_ts, [' lat', ' lon']].tolist()
        end_gps = input_df.loc[end_ts, [' lat', ' lon']].tolist()    
        start_gps_label = geolocator.reverse('{}, {}'.format(start_gps[0], start_gps[1]))
        end_gps_label = geolocator.reverse('{}, {}'.format(end_gps[0], end_gps[1]))
        
        interest_interval = input_df.loc[start_ts:end_ts].index
        loc_collector = []
        last_loc_collected_ts = 0
        travel_distance = 0
        for ii in interest_interval:
            if ii > last_loc_collected_ts+gps_list_point_min_interval:
                ii_loc = input_df.loc[ii, [' gpsts', ' lat', ' lon']].tolist()
                if len(loc_collector)>0:
                    travel_distance += haversine_formula(ii_loc[1], ii_loc[2], loc_collector[-1][1], loc_collector[-1][2])/1000
                loc_collector.append(ii_loc)
                last_loc_collected_ts = ii

        gps_data.append([start_gps_label, end_gps_label, travel_distance, start_gps, end_gps, loc_collector])
        print(int(100*i/len(start_times)),"%", end="\r")
        i+=1
        
    output_df[['start_gps_label','end_gps_label','travel_distance_km','start_gps_coord','end_gps_coord','travel_gps_list']]=np.array(gps_data,dtype=object)
    return output_df

df_car = None
# Prepare data
filenames = glob.glob('./csv/*_cardata_*.csv')
for fname in filenames:
    df2 = parse_cardata(fname, columns)
    
    if df_car is None:
        df_car = df2.copy()
    elif df2.size > 0:
        df3 = df_car.append(df2, sort=True)
        df_car = df3.copy()
        del df3
    else:
        print('no data found')
    del df2
df_car = df_car.sort_index().bfill()
df_car = df_car.groupby(level=0).last().dropna(subset=[' gpsts'])

df_car = remove_incoherent_gps_entry(df_car)

# If you want to rewrite the travel csv file the run the two commented lines
df_travel = create_travel_based_df(df_car)
df_travel.to_csv("csv/travel_based_dataframe.csv", encoding='utf-8')

df_travel = pd.read_csv("csv/travel_based_dataframe.csv",index_col=0, converters={"start_gps_coord": literal_eval, "end_gps_coord": literal_eval, "travel_gps_list": literal_eval})

## Remove travel where travel distance is lower than 500m
df_travel = df_travel[df_travel['travel_distance_km'] >0.5]

relevant_columns = ['start_hour_hmin', 'start_gps_label', 'duration_minsec', 'travel_distance_km', 'end_hour_hmin', 'end_gps_label', 'weekday_str']
df_travel.sort_values('start_hour')[relevant_columns]

print("done")