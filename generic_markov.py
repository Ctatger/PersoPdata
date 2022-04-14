# %%
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from matplotlib.pyplot import figure


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
    def __init__(self, df=None):
        """MK_chain class constructor, computes transition matrix and gamma from given dataframe

        Arguments:
            df {pandas.dataframe} -- dataframe, containing clusters ID for start and arrival
        """
        if df is None:
            self.data_frame = pd.DataFrame()
        # Stored df as class data for ease of use
        else:
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
        if self.data_frame.empty:
            self.data_frame = new
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

        else:
            start = new['Start_cluster'].values
            end = new['End_cluster'].values

            frames = [self.data_frame, new]
            self.data_frame = pd.concat(frames, ignore_index=True)

            # If new contains cluster id not existing previously, computing gamma and T_mat all over again is needed
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
            self.data_frame.reset_index(drop=True, inplace=True)

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
