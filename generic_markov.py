# %%
import pandas as pd
import numpy as np
import random
import csv

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from matplotlib.pyplot import figure
from random import randint
from ipyleaflet import Map, basemaps, basemap_to_tiles, CircleMarker
from data_parsing import parse_csv, create_window_dataframe
from IPython.display import display
from data_parsing import format_time


def evaluate_mk(df, mk):
    acc_total = 0
    for r_id in range(len(df)):

        row = df.iloc[[r_id]]
        start = row['Start_cluster'].values
        pred = mk.predict(str(start[0]))
        answ = str(row['End_cluster'].values[0])

        if pred == answ:
            acc_total += 1
    return (acc_total/len(df))*100


class generic_markov:
    def __init__(self, df):
        """MK_chain class constructor, computes transition matrix and gamma from given dataframe

        Arguments:
            df {pandas.dataframe} -- dataframe, containing clusters ID for start and arrival
        """
        # Stored df as class data for ease of use
        self.data_frame = df

        # List used to store occurences of trip scenario. Used for proba computation
        self.sum_mat = []
        # Used to check if proba in matrix add up to 1, margin for float approx
        self.TOLERANCE_VALUE = 0.0001

        # Trip clusters informations
        self.Start_clusters = self.data_frame['Start_cluster'].unique()
        self.Start_clusters.sort()
        self.End_clusters = self.data_frame['End_cluster'].unique()
        self.End_clusters.sort()

        # Computes and stores Transition matrix and gamma vector
        self.create_transition_matrix()

    def create_sum_matrix(self):
        """Parses dataset to store occurences of each start-end cluster combination
           Format : [Start, End, nb_occurence]
        """
        for Start in self.Start_clusters:
            Total = 0
            Starting_points = self.data_frame.loc[self.data_frame['Start_cluster'] == Start]

            for End in self.End_clusters:
                Ending_points = Starting_points.loc[Starting_points['End_cluster'] == End]
                Total = len(Ending_points)

                self.sum_mat.append([Start, End, Total])

    def create_proba_vector(self, Start):
        """Creates an array containing probability for each existing end cluster, given a start cluster

        Arguments:
            Start {int} -- Start cluster for which to compute proba vector

        Returns:
            np.array -- Probability vector
        """
        size = max(self.End_clusters[-1], self.Start_clusters[-1])

        # Avoid crashing in case first trip is from cluster 0 to cluster 0
        if size == 0:
            size = 1

        proba_vector = np.zeros(size+1)
        starting_points = self.data_frame.loc[self.data_frame['Start_cluster'] == Start]
        sample = len(starting_points)

        # Browse through sum_matrix (format [start_cluster, end_cluster, nb_occurence])
        for freq in self.sum_mat:
            if freq[0] == Start:
                # If start_cluster is equal to Start, computes probability
                proba_vector[freq[1]] = (freq[2]/sample)*100
        return proba_vector

    def create_gamma(self):
        """Computes gamma vector, can be used for first prediction at the start

        Raises:
            ValueError: Probability in gamma have to add up to 1 (with a tolerated margin)

        Returns:
            list -- gamma vector, chances of each start cluster to be selected as starting point of trip
        """
        size = max(self.End_clusters[-1], self.Start_clusters[-1])
        # Avoid crashing in case first trip is from cluster 0 to cluster 0
        if size == 0:
            size = 1
        sample = len(self.data_frame)

        gamma = np.zeros(size+1)

        for start in self.Start_clusters:
            df_s = self.data_frame.loc[self.data_frame['Start_cluster'] == start]
            gamma[start] = ((len(df_s)/sample)*100)
        if sum(gamma) > 100.5:
            raise ValueError("Gamma matrix coefficients not adding up to 1")
        return gamma

    def create_transition_matrix(self):
        """Computes Markov chain's transition matrix, of dimension (n*n) with n the nb of clusters.
           Each line is computed using the "create_proba_vector" method

        Returns:
            np.array -- Transition matrix of MK_chain instance
        """
        self.create_sum_matrix()

        size = max(self.End_clusters[-1], self.Start_clusters[-1])
        # Avoid crashing in case first trip is from cluster 0 to cluster 0
        if size == 0:
            size = 1
        t_mat = np.array(self.create_proba_vector(0))

        for i in range(1, size+1):
            vect = self.create_proba_vector(i)
            t_mat = np.vstack((t_mat, vect))

        self.gamma = self.create_gamma()
        self.transitionMatrix = t_mat

        self.states = [str(x) for x in range(len(self.transitionMatrix))]

        return self.gamma, t_mat

    def fit(self, new):
        """Updates transition matrix and gamma vector when new trip is given. Adds the trip to the existing database

        Arguments:
            new {pandas.dataframe} --Single entry dataframe representing one trip
        """
        start = new['Start_cluster'].values
        end = new['End_cluster'].values

        frames = [self.data_frame, new]
        self.data_frame = pd.concat(frames)

        # If new line contains cluster id not existing previously, computing gamma and T_mat all over again is needed
        if (start > self.Start_clusters[-1] or end > self.End_clusters[-1]):
            self.Start_clusters = self.data_frame['Start_cluster'].unique()
            self.Start_clusters.sort()
            self.End_clusters = self.data_frame['End_cluster'].unique()
            self.End_clusters.sort()

            self.create_transition_matrix()

        else:
            self.Start_clusters = self.data_frame['Start_cluster'].unique()
            self.Start_clusters.sort()
            self.End_clusters = self.data_frame['End_cluster'].unique()
            self.End_clusters.sort()

            for start_point in start:
                for end_point in end:
                    for freq in self.sum_mat:

                        if (freq[0] == start_point and freq[1] == end_point):
                            new_id = self.sum_mat.index(freq)
                            self.sum_mat[new_id][2] += 1

            for i in start:
                vect = self.create_proba_vector(i)
                self.transitionMatrix[i] = vect
                self.gamma = self.create_gamma()

    def predict(self, curr_st):
        """Gives id of predicted end cluster, given start of trip

        Arguments:
            curr_st {str} -- Id of current trip starting cluster

        Returns:
            str -- Predicted end cluster for trip
        """
        if curr_st in self.states:
            st_id = self.states.index(curr_st)
            most_likely = max(self.transitionMatrix[st_id])
            pred_id = np.where(self.transitionMatrix[st_id] == most_likely)[0][0]
            return self.states[pred_id]
        else:
            # If given cluster Id not existing in states, outputs -1
            return('-1')

    def gamma_predict(self):

        pred_vect = np.matmul(self.gamma, self.transitionMatrix)
        pred = np.where(pred_vect == max(pred_vect))[0][0]

        return str(pred)


# %%
if __name__ == "__main__":

    sns.set_theme()
    palette = plt.get_cmap('Set1')
    figure(figsize=(16, 14), dpi=80)

    RANGE = 100
    # df_wind = pd.DataFrame(columns=['Pos', 'Start_cluster', 'End_cluster', 'Wd_state', 'Day', 'Time', 'Time_delta'])
    window_state = []
    days = [randint(0, 6) for x in range(RANGE)]
    # possible_adresses = ["golf", "RSWL","maison st cyp", "maison cote pavee","maison saint agne"]
    possible_adresses = [[43.575319, 1.364180], [43.579300, 1.378159], [43.597517, 1.433078], [43.594339, 1.465000],
                         [43.583054, 1.450124]]
    adresses_polygon = [[43.575414, 1.364311, 43.575223, 1.364048],
                        [43.579395, 1.378290, 43.579204, 1.378027],
                        [43.597612, 1.433209, 43.597421, 1.432946],
                        [43.594434, 1.465131, 43.594243, 1.464868],
                        [43.583149, 1.450255, 43.582958, 1.449992]]

    for k in range(10):
        rand = np.random.choice(list(range(5)), RANGE, p=[0.05, 0.5, 0.1, 0.1, 0.25])
        adresses = [[random.uniform(adresses_polygon[i][2], adresses_polygon[i][0]),
                     random.uniform(adresses_polygon[i][3], adresses_polygon[i][1])]
                    for i in rand]

        Starting_hours = np.linspace(9, 10.15)
        Stopping_hours = np.linspace(17, 18.15)

        possible_times = np.concatenate([Starting_hours, Stopping_hours])
        random_times = np.random.choice(possible_times, RANGE)
        Time = []

        FMT = '%H:%M'
        for index in range(RANGE):
            Time.append(format_time(random_times[index]))
            if (rand[index] == 0 or rand[index] == 1):
                window_state.append(np.random.choice([0, 1], 1, p=[0.8, 0.2]))
            else:
                window_state.append(np.random.choice([0, 1], 1, p=[0.01, 0.99]))

        with open('/home/celadodc-rswl.com/corentin.tatger/PersoPdata/app_data/dummy_data_{}.csv'.format(k),
                  mode='w') as csv_file:
            fieldnames = ['Pos_lat', 'Pos_lon', 'Wd_state', 'Time', 'Day']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(RANGE):
                writer.writerow({'Pos_lat': adresses[i][0], 'Pos_lon': adresses[i][1], 'Wd_state': window_state[i][0],
                                 'Time': Time[i], 'Day': days[i]})

    df_csv = parse_csv(
        "/home/celadodc-rswl.com/corentin.tatger/PersoPdata/app_data/")
    df_window = create_window_dataframe(df_csv)
    # display(df_window)
    rec_colors = ['blue', 'red', 'orange', 'yellow', 'brown', 'green']
    map_layer = basemap_to_tiles(basemaps.CartoDB.Positron)
    m = Map(layers=(map_layer, ), center=((48.852, 2.246)), zoom=5, scroll_wheel_zoom=True)

    for index, row in df_window.iterrows():
        if row['Coord_cluster'] >= 0:
            m.add_layer(CircleMarker(location=row['Coordinates'], radius=3,
                                     color=rec_colors[row['Coord_cluster'] % len(rec_colors)],
                                     fill_color='#FFFFFF', weight=2))
    display(m)
    m.save('my_map.html', title='My Map')

    epoch_acc = []
    for i in range(10):
        df_window = df_window.sample(frac=1)
        df_train = df_window.sample(frac=0.7)
        df_test = df_window.drop(df_train.index)

        Mk_chain = generic_markov(df_train.head())
        current_acc = [evaluate_mk(df_test, Mk_chain)]

        for row_id in range(5, len(df_train)):
            Mk_chain.fit(df_train.iloc[[row_id]])
            current_acc.append(evaluate_mk(df_test, Mk_chain))

        epoch_acc.append(current_acc)
        print("Epoch {} done.".format(i))

    # plt.plot(list(range(30)), epoch_acc)
    # plt.plot(list(range(30)), [np.mean(epoch_acc) for i in range(30)])
    # plt.title("Mean accuracy of model is {}%".format(np.mean(epoch_acc)))
    # ax = plt.gca()
    # ax.set_ylim([60, 100])
    # plt.show()

    # Create figure
    gofig = go.Figure()
    # Add traces, one for each slider step
    for step in range(10):
        gofig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="v = " + str(step),
                x=np.arange(0, len(df_train), 1),
                y=epoch_acc[step]))
        gofig.update_yaxes(range=[0, 100])

    # Make 10th trace visible
    gofig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(gofig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(gofig.data)},
                  {"title": "Evolution of Markov Accuracy's"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    gofig.update_layout(
        sliders=sliders
    )

    gofig.show()

# %%
