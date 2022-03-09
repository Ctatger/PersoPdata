#%%

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import statistics
import pandas as pd
from ast import literal_eval
from ipyleaflet import Map, Polyline, basemaps, CircleMarker, DivIcon, Marker, DrawControl, Rectangle
import numpy as np
from sklearn import cluster
from sklearn.cluster import DBSCAN
import random
from markov import MK_chain

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
    df[header_name] = clustering_method.fit_predict(np.array([l for l in df[columns].values]))
    df["gps_start_cluster_label"] = pd.NaT
    df["gps_end_cluster_label"] = pd.NaT

    return df

def get_color(l, i, default):
    if i==-1:
        return default
    return l[i]

def color_list(data_frame,group='gps_start_cluster',seed=42,colors_arr=[]):

    n = int(np.ceil(np.power(data_frame[group].nunique(), 1/3)))
    colors = colors_arr
    for r in range(n):
        for g in range(n):
            for b in range(n):
                colors.append(color_str(int(r*255/(n-1)), int(g*255/(n-1)), int(b*255/(n-1))))
    np.random.seed(seed)#hhgg
    np.random.shuffle(colors)

    if '#ffffff' in colors:
        colors.remove("#ffffff")

    return colors

def Cluster_Rectangles(data_frame,group='gps_start_cluster'):

    cluster_rectangles=[]

    for Cluster_id in np.unique(data_frame[group]):
        if Cluster_id !=-1:
            abs_list=[]
            ord_list=[]
            bottom_left=[]
            top_right=[]
            points = data_frame.loc[data_frame[group] == Cluster_id]
            
            for _, row in points.iterrows():
                if group == 'gps_start_cluster':
                    abs_list.append(row['start_gps_coord'][0])
                    ord_list.append(row['start_gps_coord'][1])
                else:
                    abs_list.append(row['end_gps_coord'][0])
                    ord_list.append(row['end_gps_coord'][1])

            bottom_left.append(min(abs_list))
            bottom_left.append(min(ord_list))
            top_right.append(max(abs_list))
            top_right.append(max(ord_list))
            cluster_rectangles.append((tuple(bottom_left),tuple(top_right)))
    return cluster_rectangles

def Cluster_Labels(data_frame,group='gps_start_cluster'):
    for Cluster_id in np.unique(data_frame[group]):
        if Cluster_id !=-1:
            points = data_frame.loc[data_frame[group] == Cluster_id]
            if group == 'gps_start_cluster':
                data_frame.loc[data_frame[group] == Cluster_id, 'gps_start_cluster_label'] = points['start_gps_label'].mode()[0]
            else:
                data_frame.loc[data_frame[group] == Cluster_id, 'gps_end_cluster_label'] = points['end_gps_label'].mode()[0]

def Compute_Proba(data_frame,n_startcluster,n_endcluster, coeff_matrix=None):
    
    if coeff_matrix is None:
        coeff_matrix = np.ones((n_startcluster*n_endcluster,2))*1e-6
        coeff_matrix = coeff_matrix.tolist()
    
    Start_clusters=data_frame['gps_start_cluster'].unique()
    End_clusters=data_frame['gps_end_cluster'].unique()

    for k in range(len(Start_clusters)):
        Total=0
        Starting_points = data_frame.loc[data_frame['gps_start_cluster'] == Start_clusters[k]]
        for l in range(len(End_clusters)):
            Ending_points= Starting_points.loc[Starting_points['gps_end_cluster'] == End_clusters[l]]
            Total+=len(Ending_points)
            
            if  not coeff_matrix[k*len(End_clusters)+l]:
                coeff_matrix[k*len(End_clusters)+l].append(len(Starting_points))
                coeff_matrix[k*len(End_clusters)+l].append(len(Ending_points))
            else:
                coeff_matrix[k*len(End_clusters)+l][0]+= len(Starting_points)
                coeff_matrix[k*len(End_clusters)+l][1]+= len(Ending_points)

        if Total != len(Starting_points):
            raise ValueError("Probabilities not adding up to 1, check dataframe",Start_clusters[k])
    return coeff_matrix

def Create_gamma(df):
    gamma=[]
    sample=len(df)
    
    for start in df['gps_start_cluster'].unique():
        df_s=df.loc[df['gps_start_cluster'] == start]
        gamma.append((len(df_s)/sample)*100)
    if sum(gamma) > 100.5:
        raise ValueError("Gamma matrix coefficients not adding up to 1")
    return gamma

def Create_ProbabilityMatrix(data_frame,coeff_mat=None):
    T_matrix=[]
    State_prob=[]
    prob={}

    Start_clusters=data_frame['gps_start_cluster'].unique()
    End_clusters=data_frame['gps_end_cluster'].unique()

    for k in range(len(Start_clusters)):
        prob={str(x): 0 for x in Start_clusters}

        for l in range(len(End_clusters)):
            State_prob.append([End_clusters[l],coeff_mat[(k*len(End_clusters))+l][1]/coeff_mat[(k*len(End_clusters))+l][0]])
            #print("Probs for end cluster ",End_clusters[l]," ",coeff_mat[k+l])
        for key in State_prob:
            
            prob[str(key[0])] = key[1]

        T_matrix.append(list(prob.values()))
    return T_matrix
    
def Remove_redundant_travels(data_frame):
    df=data_frame.drop(data_frame[data_frame.gps_start_cluster_label == data_frame.gps_end_cluster_label].index)
    df=df.reset_index(drop=True)
    return df

def Df_Kfold(data_frame,n_fold):
    Indexes=[]
    k, m = divmod(len(data_frame), n_fold)
    for i in range(n_fold):
        Indexes.append([i*k+min(i, m),(i+1)*k+min(i+1, m)])
    return Indexes

def Evaluate_model(data_frame,M_chain,Kfold_index,n_startcluster,n_endcluster):   
    Score=[]
    for i in range(len(Kfold_index)):

        Predictions=[]
        Answers=[]
 
        Testset=data_frame[Kfold_index[i][0]:Kfold_index[i][1]]
        Trainset=Kfold_index.copy()
        del Trainset[i]

        c_mat=Compute_Proba(data_frame[Trainset[0][0]:Trainset[0][1]],n_startcluster,n_endcluster)

        for k in Trainset[1:]:
            c_mat=Compute_Proba(data_frame[k[0]:k[1]],n_startcluster,n_endcluster,c_mat)

        for _,row in Testset.iterrows():
            Cur_state=str(row['gps_start_cluster'])
            Answers.append(str(row['gps_end_cluster']))
            Predictions.append(M_chain.predict(Cur_state,filter=False))
            result=[(1) if Answers[m] == Predictions[m] else (0) for m in range(len(Predictions)) ]
        Score.append((sum(result)/len(result)*100))
        print("Precision for split {} as test split is : {}%".format(i,sum(result)/len(result)))
    return Score

#%%

if __name__ == "__main__":

    sns.set_theme()

    WEEKDAY=(0,5)
    WEEKEND=(5,7)

    HOUR_EarlyMorning=(1,6)
    HOUR_Morning=(6,9)
    HOUR_Midday=(9,16)
    HOUR_Evening=(16,19)
    HOUR_LateEvening=(19,24)
    HOUR_Afternoon=(12,23)
    
    KFOLD=6

    df_travel = pd.read_csv("csv/travel_based_dataframe.csv",index_col=0, converters={"start_gps_coord": literal_eval, "end_gps_coord": literal_eval, "travel_gps_list": literal_eval})

    ## Remove travel where travel distance is lower than 500m
    df_travel = df_travel[df_travel['travel_distance_km'] >0.5]

    relevant_columns = ['start_hour_hmin', 'start_gps_label', 'duration_minsec', 'travel_distance_km', 'end_hour_hmin', 'end_gps_label', 'weekday_str']
    #df_travel.sort_values('start_hour')[relevant_columns]

    trip_to_track = [28]
    m = prepare_map(df_travel, trip_to_track)

    #Classifie les trajets et ajoute une colonne avec l'ID cluster à la df
    dbscan = DBSCAN(eps=0.005, min_samples=3)
    create_clusters(df_travel, 'start_gps_coord', dbscan, header_name='gps_start_cluster')

    #Classifie les trajets et ajoute une colonne avec l'ID cluster à la df
    dbscan = DBSCAN(eps=0.005, min_samples=3)
    create_clusters(df_travel, 'end_gps_coord', dbscan, header_name='gps_end_cluster')

    df_travel = df_travel[df_travel['gps_end_cluster'] != -1]

    start_rec=Cluster_Rectangles(df_travel)
    end_rec=Cluster_Rectangles(df_travel,group='gps_end_cluster')
    Cluster_Labels(df_travel)
    Cluster_Labels(df_travel,group='gps_end_cluster')

    m = Map(basemap=basemaps.OpenStreetMap.France, zoom=9, scroll_wheel_zoom=True)   
    vals = [l for l in df_travel['start_gps_coord'].values]
    ctr = np.mean(vals, axis=0)
    m.center = list(ctr)

    for i, coord in enumerate([l for l in df_travel['start_gps_coord'].values]):
        m.add_layer(CircleMarker(location=list(coord), radius=3, color="#0000FF", fill_color='#FFFFFF',weight=2))

    for j, coord in enumerate([l for l in df_travel['end_gps_coord'].values]):
        m.add_layer(CircleMarker(location=list(coord), radius=3, color="#FF0000", fill_color='#FFFFFF',weight=2))

    for R_start in start_rec:
        rectangle_start = Rectangle(bounds=R_start,weight=2)
        m.add_layer(rectangle_start)
    for R_end in end_rec:
        rectangle_end = Rectangle(bounds=R_end,weight=2,color="#FF0000")
        m.add_layer(rectangle_end)

    m.save('my_map.html', title='My Map')


    df_travel=Remove_redundant_travels(df_travel)

    st_cluster=len(df_travel.gps_start_cluster.unique())
    end_cluster=len(df_travel.gps_end_cluster.unique())

    Kfold_index=Df_Kfold(df_travel,KFOLD)
    C_mat=Compute_Proba(df_travel,st_cluster,end_cluster)
    TMat=Create_ProbabilityMatrix(df_travel,C_mat)

    M_chain=MK_chain(TMat)

    Scores=Evaluate_model(df_travel,M_chain,Kfold_index,st_cluster,end_cluster)

    figure(figsize=(16, 14), dpi=80)
    #graph of model's results
    plt.scatter(list(range(KFOLD)),Scores)

    plt.axhline(y=statistics.median(Scores), color='r', linestyle='-')
    plt.title("Median Acurracy with for Kfold training: {:5.2f}%".format(statistics.median(Scores)))

    plt.show()

    #graph of arrival point repartition
    aa=df_travel[df_travel['gps_start_cluster'] == 0]
    a = pd.DataFrame({ 'group' : np.repeat('0',len(aa)), 'value': aa['gps_end_cluster'] })
    bb=df_travel[df_travel['gps_start_cluster'] == 1]
    b = pd.DataFrame({ 'group' : np.repeat('1',len(bb)), 'value': bb['gps_end_cluster'] })
    cc=df_travel[df_travel['gps_start_cluster'] == 2]
    c = pd.DataFrame({ 'group' : np.repeat('2',len(cc)), 'value': cc['gps_end_cluster'] })
    dd=df_travel[df_travel['gps_start_cluster'] == 3]
    d = pd.DataFrame({ 'group' : np.repeat('3',len(dd)), 'value': dd['gps_end_cluster'] })
    ee=df_travel[df_travel['gps_start_cluster'] == 4]
    e = pd.DataFrame({ 'group' : np.repeat('4',len(ee)), 'value': ee['gps_end_cluster'] })

    df=a.append(b).append(c).append(d).append(e)

    # plot violin chart
    ax = sns.violinplot( x='group', y='value', data=df)
    ax = sns.stripplot(x='group', y='value', data=df, color="orange", jitter=0.2, size=2.5)
    plt.xlabel("Starting point")
    plt.ylabel("Ending point")


    # add title
    plt.title("Ending cluster distribution")
    # show the graph
    plt.show()
#%%