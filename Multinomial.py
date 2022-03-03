#%%
from nis import match
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ast import literal_eval
from travel_clustering import create_clusters,Cluster_Labels
from sklearn.cluster import DBSCAN

import math as mt
#%%
def Time_slot(df):
    HOUR_EarlyMorning=(0,7)
    HOUR_Morning=(7,10)  
    HOUR_Midday=(10,16)
    HOUR_Evening=(16,19)
    HOUR_LateEvening=(19,24)
    #nuit a add
    
    df['time_slot']=pd.NaT

    slot={"HOUR_EarlyMorning":(0,7),"HOUR_Morning":(7,11),"HOUR_Midday":(11,17),"HOUR_Evening":(17,20),"HOUR_LateEvening":(20,24)}

    for idc,rows in df.iterrows():
        for i in slot.values():
            hour=rows['start_hour_hmin'].split(":")
            if int(hour[0]) in(list(range (i[0],i[1]))):
                df.loc[idc,'time_slot']=list(slot.keys())[list(slot.values()).index(i)]
    return df


def standardScaler(feature_array):
    """Takes the numpy.ndarray object containing the features and performs standardization on the matrix.
    The function iterates through each column and performs scaling on them individually.
    
    Args-
        feature_array- Numpy array containing training features
    
    Returns-
        None
    """
    
    total_cols = feature_array.shape[1] # total number of columns 
    for i in range(total_cols): # iterating through each column
        feature_col = feature_array[:, i]
        mean = feature_col.mean() # mean stores mean value for the column
        std = feature_col.std() # std stores standard deviation value for the column
        feature_array[:, i] = (feature_array[:, i] - mean) / std # standard scaling of each element of the column

def compute_daybased(df,weekend=False):
    Start_clusters=df['gps_start_cluster'].unique()
    if weekend:
        df=df.loc[(df['weekday']==5) | (df['weekday']==6)]
    df_ts=Time_slot(df)
    

    P_daybased = []

    for k in range(len(Start_clusters)):
        Starting_points = df.loc[df['gps_start_cluster'] == Start_clusters[k]]
        P_daybased.append((len(Starting_points)/len(df))*100)
    return P_daybased

def compute_weekbased(df,weekend=False):
    pass

def fit():
    pass

def predict(Vin_array):
    """Computes the probability for each destination to be selected wand returns the most likely pick

    Arguments:
        Vin_array {array} -- Contains deterministic components for each cluster

    Returns:
        string -- Id of predicted arrival cluster
    """
    Pin_array=[]
    for Vin in Vin_array:
        Pin=mt.exp(Vin)/(sum((mt.exp(Vjn) for Vjn in Vin_array)))
        Pin_array.append(Pin)
    return str(Pin_array.index(max(Pin_array)))
   



figure(figsize=(1,1))

df_travel = pd.read_csv("csv/travel_based_dataframe.csv",index_col=0, converters={"start_gps_coord": literal_eval, "end_gps_coord": literal_eval, "travel_gps_list": literal_eval})
df_travel = df_travel[df_travel['travel_distance_km'] >0.5]

P_markov=[]
P_daybased_weekday=[]
P_daybased_weekend=[]
P_weekbased_weekday=[]
P_weekbased_weekend=[]


features_weekday=[P_markov,P_daybased_weekday,P_weekbased_weekday]
features_weekend=[P_markov,P_daybased_weekend,P_weekbased_weekend]
features_weekday=np.array(features_weekday)
features_weekend=np.array(features_weekend)

dbscan = DBSCAN(eps=0.005, min_samples=3)
create_clusters(df_travel, 'start_gps_coord', dbscan, header_name='gps_start_cluster')

#Classifie les trajets et ajoute une colonne avec l'ID cluster à la df
dbscan = DBSCAN(eps=0.005, min_samples=3)
create_clusters(df_travel, 'end_gps_coord', dbscan, header_name='gps_end_cluster')
Cluster_Labels(df_travel)

target=df_travel['gps_end_cluster']
#%%