# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


import math as mt

from matplotlib.pyplot import figure
from ast import literal_eval
from travel_clustering import create_clusters, Cluster_Labels,\
    Compute_Proba, Create_ProbabilityMatrix, Create_gamma, \
    Remove_redundant_travels
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from markov import MK_chain
from data_parsing import create_dataframe

# default='warn', disables annoying warning
pd.options.mode.chained_assignment = None

# %%


def Time_Slot(df):
    """Parse dataframe to assign a time slot to each trip's start time

    Arguments:
        df {pandas.dataframe} -- dataframe, containing clusters ID for start
                                    and arrival

    Returns:
        pandas.dataframe -- Input dataframe with new 'time_slot' column added
    """
    df.loc[:, 'time_slot'] = pd.NaT

    slot = {"HOUR_EarlyMorning": (0, 7), "HOUR_Morning": (7, 11),
            "HOUR_Midday": (11, 17), "HOUR_Evening": (17, 20),
            "HOUR_LateEvening": (20, 24)}

    for idc, rows in df.iterrows():
        for i in slot.values():
            hour = rows['start_hour_hmin'].split(":")
            if int(hour[0]) in (list(range(i[0], i[1]))):
                t_slot = list(slot.keys())[list(slot.values()).index(i)]
                df.loc[idc, 'time_slot'] = t_slot
    return df


def Standard_Scaler(feature_array):
    """Takes the numpy.ndarray object containing the features and performs
    standardization on the matrix.The function iterates through each column
    and performs scaling on them individually.

    Args-
        feature_array- Numpy array containing training features

    Returns-
        None
    """

    total_cols = feature_array.shape[1]  # total number of columns
    for i in range(total_cols):  # iterating through each column
        feature_col = feature_array[:, i]
        mean = feature_col.mean()  # mean stores mean value for the column
        # std stores standard deviation value for the column
        std = feature_col.std()
        # standard scaling of each element of the column
        feature_array[:, i] = (feature_array[:, i] - mean) / std


def Compute_Weekbased(df):
    """Parse dataframe to process the probabilities based on time of the week

    Arguments:
        df {pandas.dataframe} -- dataframe, containing clusters ID
                                    for start and arrival

    Returns:
        Dict -- Key: weekday/weekend
                    Value: Probability for each cluster to be selected
    """
    end_cluster = df.gps_end_cluster.unique()

    P_weekbased = {"weekday": {}, "weekend": {}}

    df_w = df.loc[(df['weekday'] == 5) | (df['weekday'] == 6)]
    df_ts = Time_Slot(df_w)
    P_day = {}

    for slots in df_ts['time_slot'].unique():
        P_end = {}
        df_slot = df_ts.loc[df_ts['time_slot'] == slots]
        for end in end_cluster:
            df_prob = df_slot.loc[df_slot['gps_end_cluster'] == end]
            prob = (len(df_prob) / len(df_slot)) * 100
            P_end[end] = prob
        P_day[slots] = P_end
        P_weekbased["weekend"] = P_day

    df_w = df.loc[(df['weekday'] != 5) & (df['weekday'] != 6)]
    df_ts = Time_Slot(df_w)
    P_day = {}

    for slots in df_ts['time_slot'].unique():
        P_end = {}
        df_slot = df_ts.loc[df_ts['time_slot'] == slots]
        for end in end_cluster:
            df_prob = df_slot.loc[df_slot['gps_end_cluster'] == end]
            prob = (len(df_prob) / len(df_slot)) * 100
            P_end[end] = prob
        P_day[slots] = P_end
        P_weekbased["weekday"] = P_day

    return P_weekbased


def Compute_Daybased(df):
    """Parse dataframe to process the probabilities based on time slot

    Arguments:
        df {pandas.dataframe} -- dataframe, containing clusters ID
                                    for start and arrival

    Returns:
        Dict -- Key: Day of the week &
                    Value: Probability for each cluster to be selected
                    depending on time slot
    """

    end_cluster = df.gps_end_cluster.unique()

    P_daybased = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}

    for day in df['weekday'].unique():
        df_d = df.loc[df['weekday'] == day]
        df_ts = Time_Slot(df_d)

        P_day = {}

        for slots in df_ts['time_slot'].unique():
            P_start = {}
            df_slot = df_ts.loc[df_ts['time_slot'] == slots]

            for end in end_cluster:
                df_prob = df_slot.loc[df_slot['gps_end_cluster'] == end]
                prob = (len(df_prob) / len(df_slot)) * 100
                P_start[end] = prob
            P_day[slots] = P_start
        P_daybased[day] = P_day
    return P_daybased


def Compute_Markov(df):
    """Processes dataframe to create intitial state and
        transition matrix of markov chain

    Arguments:
        df {pandas.dataframe} -- dataframe, containing clusters ID
                                    for start and arrival

    Returns:
        (list,list) -- Initial state matrix (gamma) and
                        Transition state matrix(TMat)
    """
    st_cluster = len(df.gps_start_cluster.unique())
    end_cluster = len(df.gps_end_cluster.unique())

    C_mat = Compute_Proba(df, st_cluster, end_cluster)
    TMat = Create_ProbabilityMatrix(df, C_mat)
    gamma = Create_gamma(df)

    return gamma, TMat


def predict_trip(df, day, slot, P_type='day'):
    maxi = 0
    maxi_id = 0
    if P_type == 'day':
        pred = Compute_Daybased(df)
        for i in df.gps_end_cluster.unique():
            if pred[day][slot][i] > maxi:
                maxi = pred[day][slot][i]
                maxi_id = i
        return maxi_id
    if P_type == 'week':
        pred = Compute_Weekbased(df)
        if day > 4:
            part = 'weekdend'
        else:
            part = 'weekday'

        for i in df.gps_end_cluster.unique():
            if pred[part][slot][i] > maxi:
                maxi = pred[part][slot][i]
                maxi_id = i
    return maxi_id


def fit_trip(daybased, weekbased, markov, day, t_slot, end_cluster):
    """const = 0
    alpha, beta, delta = (0,) * 3

    if day in list(range(5)):
        weekpart = "weekday"
    else:
        weekpart = "weekend"

    Prob = const + alpha * weekbased[weekpart][t_slot][end_cluster]
            + beta * daybased[day][t_slot][
        end_cluster] + delta * max(markov)"""


def fit(df, daybased, weekbased, markov):
    Coefficients = []

    for day in range(7):
        for slot in df['time_slot'].unique():
            for end in df.gps_end_cluster.unique():
                Coefficients.append(fit_trip(daybased, weekbased,
                                    markov, slot, end))

    alpha = np.mean([i[0] for i in Coefficients])
    beta = np.mean([i[1] for i in Coefficients])
    delta = np.mean([i[2] for i in Coefficients])
    return alpha, beta, delta


def predict(Vin_array):
    """Computes the probability for each destination to be selected
        and returns the most likely pick

    Arguments:
        Vin_array {array} -- Contains deterministic components for each cluster

    Returns:
        string -- Id of predicted arrival cluster
    """
    Pin_array = []
    for Vin in Vin_array:
        Pin = mt.exp(Vin) / (sum((mt.exp(Vjn) for Vjn in Vin_array)))
        Pin_array.append(Pin)
    return str(Pin_array.index(max(Pin_array)))


def Graph_Cluster(Mat):
    fig, axs = plt.subplots(3, 3, figsize=(15, 6),
                            facecolor='w', edgecolor='k')

    fig.subplots_adjust(hspace=.5, wspace=.001)

    axs = axs.ravel()

    for i in range(len(Mat)):
        axs[i].plot(list(range(-1, len(T_markov) - 1)), Mat[i])


if __name__ == "__main__":

    # Sets the graphical theme as seaborn default
    sns.set_theme()
    figure(figsize=(1, 1))

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

    noise_ID = min(df_travel['gps_end_cluster'].unique())

    df_travel = df_travel[df_travel['gps_end_cluster'] != noise_ID]
    df_travel = df_travel[df_travel['gps_start_cluster'] != noise_ID]

    # Adds location label, using gps street data
    Cluster_Labels(df_travel)
    df_travel = Remove_redundant_travels(df_travel)

    # Markov chain creation
    g, T_markov = Compute_Markov(df_travel)
    M_chain = MK_chain(T_markov)

    P_markov = np.matmul(g, T_markov)
    # Creating a list of predictions for each trip
    p_markov = np.ones((len(df_travel)), dtype=int)*np.argmax(P_markov)
    p_week = np.ones((len(df_travel)), dtype=int)*predict_trip(df_travel, 3,
                                                               'HOUR_Morning',
                                                               P_type='week')
    p_day = np.ones((len(df_travel)), dtype=int)*predict_trip(df_travel, 3,
                                                              'HOUR_Morning',
                                                              P_type='day')

    d = {"Markov": p_markov, "Week": p_week, "Day": p_day}
    mle_dataset = pd.DataFrame(data=d)
    preds = mle_dataset[['Markov', 'Week', 'Day']]
    ground_truth = df_travel['gps_end_cluster'].values
    # Initial analysis of the model's parameters
    # model = sm.OLS(Ground_truth, Preds.astype(float))
    # results = model.fit()

    mlr = LinearRegression()
    mlr.fit(preds, ground_truth)

    # y_pred = np.around(mlr.predict(x_test))
    # %%
