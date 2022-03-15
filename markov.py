# %%
import matplotlib.pyplot as plt
from math import gamma
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


class MK_chain:
    def __init__(self, df):
        """__init__ Markov chain Class constructor

        Arguments:
            Prob_mat  -- Transition matrix of the markov chain

        Raises:
            ValueError: Sum of probibilities in a given row should always add up to 1
        """
        self.data_frame = df

        self.trips_nb = []
        self.Start_clusters = []
        self.End_clusters = []
        self.sum_mat = []
        self.TOLERANCE_VALUE = 0.0001

        self.Start_clusters = self.data_frame['gps_start_cluster'].unique()
        self.Start_clusters.sort()
        self.End_clusters = self.data_frame['gps_end_cluster'].unique()
        self.End_clusters.sort()

        self.create_transition_matrix()

    def create_sum_matrix(self):

        for Start in self.Start_clusters:
            Total = 0
            Starting_points = self.data_frame.loc[self.data_frame['gps_start_cluster'] == Start]

            for End in self.End_clusters:
                Ending_points = Starting_points.loc[Starting_points['gps_end_cluster'] == End]
                Total = len(Ending_points)

                self.sum_mat.append([Start, End, Total])

    def create_proba_vector(self, Start):

        size = max(self.End_clusters[-1], self.Start_clusters[-1])
        if size == 1:
            size = 1

        proba_vector = np.zeros(size+1)
        starting_points = self.data_frame.loc[self.data_frame['gps_start_cluster'] == Start]
        sample = len(starting_points)

        for freq in self.sum_mat:
            if freq[0] == Start:
                proba_vector[freq[1]] = (freq[2]/sample)*100
        return proba_vector

    def create_gamma(self):
        gamma = []
        sample = len(self.data_frame)

        for start in self.Start_clusters:
            df_s = self.data_frame.loc[self.data_frame['gps_start_cluster'] == start]
            gamma.append((len(df_s)/sample)*100)
        if sum(gamma) > 100.5:
            raise ValueError("Gamma matrix coefficients not adding up to 1")
        return gamma

    def create_transition_matrix(self):

        self.create_sum_matrix()

        size = max(self.End_clusters[-1], self.Start_clusters[-1])
        if size == 0:
            size = 1
        t_mat = np.array(self.create_proba_vector(0))

        for i in range(1, size+1):
            vect = self.create_proba_vector(i)
            t_mat = np.vstack((t_mat, vect))

        self.gamma = self.create_gamma()
        self.transitionMatrix = t_mat

        self.states = [str(x) for x in range(len(self.transitionMatrix))]

        return gamma, t_mat

    def fit(self, new):

        start = new['gps_start_cluster'].values
        end = new['gps_end_cluster'].values

        frames = [self.data_frame, new]
        self.data_frame = pd.concat(frames)

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

        if curr_st in self.states:
            st_id = self.states.index(curr_st)
            most_likely = max(self.transitionMatrix[st_id])
            pred_id = np.where(self.transitionMatrix[st_id] == most_likely)[0][0]
            return self.states[pred_id]
        else:
            return('-1')


# %%


if __name__ == "__main__":

    sns.set_theme()
    palette = plt.get_cmap('Set1')

    epoch_acc = []
    figure(figsize=(16, 14), dpi=80)
    for i in range(10):

        df_travel = create_dataframe()
        df_travel = df_travel.sample(frac=1)

        df_data = df_travel[:1]
        df_train = df_travel[1:50]
        df_test = df_travel[50:]

        mk_travel = MK_chain(df=df_data)
        model_acc = []

        trainset_lenghts = list(range(1, 50))

        for r_id in range(len(df_train)):
            mk_travel.fit(df_train.iloc[[r_id]])
            model_acc.append(evaluate_mk(df_test, mk_travel))

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

    plt.show()


# %%
