#%%
import pandas as pd
from ast import literal_eval
from ipyleaflet import Map, Polyline, basemaps, CircleMarker, DivIcon, Marker, DrawControl, Rectangle
import numpy as np
from sklearn.cluster import DBSCAN

def color_str(r,g,b):
    return '#'+'0x{:02x}'.format(min(r,255))[2:]+'0x{:02x}'.format(min(g,255))[2:]+'0x{:02x}'.format(min(b,255))[2:]


def SquareMarker(location, length, color, fill_color, fill_opacity):
    icon = DivIcon(html='<div style="width:100%; height:100%; background-color:{}; border: 3px solid {}; opacity:{}"></div>'.format(fill_color, color, fill_opacity), bg_pos=[0, 0], icon_size=[2*length, 2*length])
    return Marker(location=location, icon=icon)    



def plot_start_end_pos(m, df, hour_interval, weekday_interval, start_or_end='start'):
    h1, h2 = hour_interval
    w1, w2 =weekday_interval
    vals = df.query('{0}_hour >= {1} & {0}_hour < {2} & weekday >= {3} & weekday < {4}'.format(start_or_end, h1, h2, w1, w2))['travel_gps_list'].values
    
    start_pos = [l[0] for l in vals]
    end_pos = [l[-1] for l in vals]
    ctr = np.mean([t[1:] for t in start_pos], axis=0)
    m.center = list(ctr)

    #Colors
    n = int(np.ceil(np.power(len(start_pos), 1/3)))
    colors = []
    for r in range(n):
        for g in range(n):
            for b in range(n):
                colors.append(color_str(int(r*255/(n)), int(g*255/(n)), int(b*255/(n))))
    np.random.seed(42)
    np.random.shuffle(colors)
    for s,e in zip(start_pos, end_pos) :
        c=colors.pop()
        m.add_layer(SquareMarker(location=list(s[1:]), length=10, color=c, fill_color=color_str(255,255,255), fill_opacity=0.8))
        m.add_layer(CircleMarker(location=list(e[1:]), radius=10, color=c, fill_color=color_str(255,255,255), fill_opacity=0.8))


def prepare_map(df, trip_to_track):
    
    #Colors
    n = int(np.ceil(np.power(len(trip_to_track), 1/3)))
    colors = []
    for r in range(n):
        for g in range(n):
            for b in range(n):
                colors.append(color_str(int(r*255/n), int(g*255/n), int(b*255/n)))
    np.random.seed(42)
    np.random.shuffle(colors)
    
    track = {}
    for k in trip_to_track:
        track[k]=df.loc[k, 'travel_gps_list']
    ctr = np.mean([t[1:] for t in track[trip_to_track[0]]], axis=0)

    m = Map(basemap=basemaps.OpenStreetMap.France, center=list(ctr), zoom=11, scroll_wheel_zoom=True)
    t_line = {}
    track_line = {}
    for k in trip_to_track:
        t_line[k] = [t[1:] for t in track[k]]
        track_line[k] = Polyline(
            locations=t_line[k],
            color=colors.pop(0) ,
            fill=False,
            weight=3
        )
        m.add_layer(track_line[k])
    return m


def create_clusters(df, columns, clustering_method, header_name=None):
    if not header_name:
        header_name="_".join(columns)
    df[header_name] = clustering_method.fit_predict(np.array([l for l in df_travel[columns].values]))
    df["gps_cluster_label"] = pd.NaT

    return df

def get_color(l, i, default):
    if i==-1:
        return default
    return l[i]

def Cluster_Rectangles(data_frame):

    cluster_rectangles=[]

    for Cluster_id in np.unique(data_frame['gps_cluster']):
        if Cluster_id !=-1:
            abs_list=[]
            ord_list=[]
            bottom_left=[]
            top_right=[]
            points = data_frame.loc[data_frame['gps_cluster'] == Cluster_id]
            
            for _, row in points.iterrows():
                abs_list.append(row['start_gps_coord'][0])
                ord_list.append(row['start_gps_coord'][1])
            
            bottom_left.append(min(abs_list))
            bottom_left.append(min(ord_list))
            top_right.append(max(abs_list))
            top_right.append(max(ord_list))
            cluster_rectangles.append((tuple(bottom_left),tuple(top_right)))
    return cluster_rectangles

def Cluster_Labels(data_frame):
    for Cluster_id in np.unique(data_frame['gps_cluster']):
        if Cluster_id !=-1:

            points = data_frame.loc[data_frame['gps_cluster'] == Cluster_id]
            print(points['start_gps_label'].mode()[0])
            data_frame.loc[data_frame["gps_cluster"] == Cluster_id, "gps_cluster_label"] = points['start_gps_label'].mode()[0]


#%%

if __name__ == "__main__":

    WEEKDAY_MonTueWed=(0,3)
    WEEKDAY_MonTueWedThu=(0,4)
    WEEKDAY_MonTueWedThuFri=(0,5)
    WEEKDAY_Weekend=(5,7)
    WEEKDAY_Mon=(0,1)
    WEEKDAY_Tue=(1,2)
    WEEKDAY_Wed=(2,3)
    WEEKDAY_Thu=(3,4)
    WEEKDAY_Fri=(4,5)
    WEEKDAY_Sat=(5,6)
    WEEKDAY_Sun=(6,7)

    HOUR_Morning=(6,12)
    HOUR_Afternoon=(12,23)
    HOUR_6_8=(6, 8)
    HOUR_8_10=(8, 10)
    HOUR_10_12=(10, 12)
    HOUR_12_14=(12, 14)
    HOUR_14_16=(14, 16)
    HOUR_16_18=(16, 18)
    HOUR_18_20=(18, 20)
    HOUR_20_22=(20, 22)
    HOUR_22_00=(22, 24)
    

    df_travel = pd.read_csv("csv/travel_based_dataframe.csv",index_col=0, converters={"start_gps_coord": literal_eval, "end_gps_coord": literal_eval, "travel_gps_list": literal_eval})

    ## Remove travel where travel distance is lower than 500m
    df_travel = df_travel[df_travel['travel_distance_km'] >0.5]

    relevant_columns = ['start_hour_hmin', 'start_gps_label', 'duration_minsec', 'travel_distance_km', 'end_hour_hmin', 'end_gps_label', 'weekday_str']
    #df_travel.sort_values('start_hour')[relevant_columns]

    trip_to_track = [28]
    m = prepare_map(df_travel, trip_to_track)
    #m

    dbscan = DBSCAN(eps=0.005, min_samples=3)
    create_clusters(df_travel, 'start_gps_coord', dbscan, header_name='gps_cluster')#Classifie les trajets et ajoute une colonne avec l'ID cluster à la df

    C_rec=Cluster_Rectangles(df_travel)
    Cluster_Labels(df_travel)

    m = Map(basemap=basemaps.OpenStreetMap.France, zoom=9, scroll_wheel_zoom=True)   
    m
    vals = [l for l in df_travel['start_gps_coord'].values]
    ctr = np.mean(vals, axis=0)
    m.center = list(ctr)
    
    #Colors
    n = int(np.ceil(np.power(df_travel['gps_cluster'].nunique(), 1/3)))
    colors = []
    for r in range(n):
        for g in range(n):
            for b in range(n):
                colors.append(color_str(int(r*255/(n-1)), int(g*255/(n-1)), int(b*255/(n-1))))
    np.random.seed(42)#hhgg
    np.random.shuffle(colors)

    if '#ffffff' in colors:
        colors.remove("#ffffff")

    clist = [get_color(colors, i,'#ffffff') for i in df_travel['gps_cluster'].values] 

    for i, coord in enumerate(vals):
        m.add_layer(CircleMarker(location=list(coord), radius=3, color=clist[i], fill_color='#FFFFFF',weight=2))
    m

    for coord in C_rec:
        rectangle = Rectangle(bounds=coord,weight=2)
        m.add_layer(rectangle)

    m
#%%