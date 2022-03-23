# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from data_parsing import create_dataframe
from matplotlib.pyplot import figure


def evaluate_mk(df, mk):
    acc_total = 0
    for r_id in range(len(df)):

        row = df.iloc[[r_id]]
        start = row['gps_start_cluster'].values
        pred = mk.predict(str(start[0]))
        answ = str(row['gps_end_cluster'].values[0])

        if pred == answ:
            acc_total += 1
    return (acc_total/len(df))*100


def evaluate_gamma_mk(df, mk):
    acc_total = 0
    for r_id in range(len(df)):

        row = df.iloc[[r_id]]
        pred = mk.gamma_predict()
        answ = str(row['gps_end_cluster'].values[0])

        if pred == answ:
            acc_total += 1
    return (acc_total/len(df))*100


class MK_chain:
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
        self.Start_clusters = self.data_frame['gps_start_cluster'].unique()
        self.Start_clusters.sort()
        self.End_clusters = self.data_frame['gps_end_cluster'].unique()
        self.End_clusters.sort()

        # Computes and stores Transition matrix and gamma vector
        self.create_transition_matrix()

    def create_sum_matrix(self):
        """Parses dataset to store occurences of each start-end cluster combination
           Format : [Start, End, nb_occurence]
        """
        for Start in self.Start_clusters:
            Total = 0
            Starting_points = self.data_frame.loc[self.data_frame['gps_start_cluster'] == Start]

            for End in self.End_clusters:
                Ending_points = Starting_points.loc[Starting_points['gps_end_cluster'] == End]
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
        starting_points = self.data_frame.loc[self.data_frame['gps_start_cluster'] == Start]
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
            df_s = self.data_frame.loc[self.data_frame['gps_start_cluster'] == start]
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
        start = new['gps_start_cluster'].values
        end = new['gps_end_cluster'].values

        frames = [self.data_frame, new]
        self.data_frame = pd.concat(frames)

        # If new line contains cluster id not existing previously, computing gamma and T_mat all over again is needed
        if (start > self.Start_clusters[-1] or end > self.End_clusters[-1]):
            self.Start_clusters = self.data_frame['gps_start_cluster'].unique()
            self.Start_clusters.sort()
            self.End_clusters = self.data_frame['gps_end_cluster'].unique()
            self.End_clusters.sort()

            self.create_transition_matrix()

        else:
            self.Start_clusters = self.data_frame['gps_start_cluster'].unique()
            self.Start_clusters.sort()
            self.End_clusters = self.data_frame['gps_end_cluster'].unique()
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

    epoch_acc = []
    gamma_epoch_acc = []

    figure(figsize=(16, 14), dpi=80)
    for i in range(10):

        df_travel = create_dataframe()
        df_travel = df_travel.sample(frac=1)

        df_data = df_travel[:1]
        df_train = df_travel[1:50]
        df_test = df_travel[50:]

        mk_travel = MK_chain(df=df_data)
        model_acc = []
        gamma_model_acc = []

        trainset_lenghts = list(range(1, 50))

        for r_id in range(len(df_train)):
            mk_travel.fit(df_train.iloc[[r_id]])
            model_acc.append(evaluate_mk(df_test, mk_travel))
            gamma_model_acc.append(evaluate_gamma_mk(df_test, mk_travel))

        gamma_epoch_acc.append(gamma_model_acc)
        epoch_acc.append(model_acc)
        print("Epoch {} done.".format(i))

    for k in range(1, 10):
        plt.subplot(3, 3, k)
        for epoch in range(1, 10):
            plt.plot(trainset_lenghts, epoch_acc[epoch], marker='', color='grey', linewidth=0.6, alpha=0.3)
        plt.plot(trainset_lenghts, epoch_acc[k], marker='', color=palette(k), linewidth=2.4, alpha=0.9)

        y_ticks = np.arange(0, 101, 25)
        x_ticks = np.arange(0, 51, 10)
        ax = plt.gca()
        # Not ticks everywhere
        if k not in [7, 8, 9]:
            ax.set_xticklabels([])
        if k not in [1, 4, 7]:
            ax.set_yticklabels([])

        ax.set_ylim([0, 100])
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
    plt.title("Evolution of accuracy without gamma consideration")
    plt.show()

    figure(figsize=(16, 14), dpi=80)

    for k in range(1, 10):
        plt.subplot(3, 3, k)
        for epoch in range(1, 10):
            plt.plot(trainset_lenghts, gamma_epoch_acc[epoch], marker='', color='grey', linewidth=0.6, alpha=0.3)
        plt.plot(trainset_lenghts, gamma_epoch_acc[k], marker='', color=palette(k), linewidth=2.4, alpha=0.9)

        y_ticks = np.arange(0, 101, 25)
        x_ticks = np.arange(0, 51, 10)
        ax = plt.gca()
        # Not ticks everywhere
        if k not in [7, 8, 9]:
            ax.set_xticklabels([])
        if k not in [1, 4, 7]:
            ax.set_yticklabels([])

        ax.set_ylim([0, 100])
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
    plt.title("Evolution of accuracy with gamma")
    plt.show()


# %%
