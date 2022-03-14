# %%
# import matplotlib.pyplot as plt
from math import gamma
import seaborn as sns
import numpy as np
import pandas as pd

from data_parsing import create_dataframe


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

    def predict(self, curr_st):

        if curr_st in self.states:
            st_id = self.states.index(curr_st)
            most_likely = max(self.transitionMatrix[st_id])
            pred_id = np.where(self.transitionMatrix[st_id] == most_likely)[0][0]
            return self.states[pred_id]
        else:
            return('-1')

    def update_transmat(self, st, prob_list):
        if abs(1-sum(prob_list)) > self.TOLERANCE_VALUE:
            raise ValueError("New probabilities don't add up to 1 (please keep the 10e-3 threshold in mind)")

        state_id = self.states.index(st)
        self.transitionMatrix[state_id] = prob_list

    def add_state(self, new_state, new_probs, old_probs):

        for k in old_probs:
            if abs(1-sum(k)) > self.TOLERANCE_VALUE:
                raise ValueError("New probabilities don't add up to 1 (please keep the 10e-3 threshold in mind)")

        if abs(1-sum(new_probs)) > self.TOLERANCE_VALUE:
            raise ValueError("Probabilities for new state don't add up to 1 (please keep the 10e-3 threshold in mind)")

        for i in range(len(self.states)):
            self.transitionMatrix[i] = old_probs[i]

        self.states.append(new_state)
        self.transitionMatrix.append(new_probs)

    def get_transmat(self):
        return self.transitionMatrix

    def get_states(self):
        return self.states

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
        proba_vector = np.zeros(size+1)
        starting_points = self.data_frame.loc[self.data_frame['gps_start_cluster'] == Start]
        sample = len(starting_points)

        for freq in self.sum_mat:
            if freq[0] == Start:
                proba_vector[freq[1]] = (freq[2]/sample)*100
        return proba_vector

    def create_transition_matrix(self):

        self.create_sum_matrix()

        size = max(self.End_clusters[-1], self.Start_clusters[-1])
        t_mat = np.array(self.create_proba_vector(0))

        for i in range(1, size+1):
            vect = self.create_proba_vector(i)
            t_mat = np.vstack((t_mat, vect))

        self.gamma = self.create_gamma()
        self.transitionMatrix = t_mat

        self.states = [str(x) for x in range(len(self.transitionMatrix))]

        return gamma, t_mat

    def create_gamma(self):
        gamma = []
        sample = len(self.data_frame)

        for start in self.Start_clusters:
            df_s = self.data_frame.loc[self.data_frame['gps_start_cluster'] == start]
            gamma.append((len(df_s)/sample)*100)
        if sum(gamma) > 100.5:
            raise ValueError("Gamma matrix coefficients not adding up to 1")
        return gamma

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


# %%


if __name__ == "__main__":

    sns.set_theme()

    epoch_acc = []
    for epoch in range(5):
        df_travel = create_dataframe()
        df_travel = df_travel.sample(frac=1)

        df_data = df_travel[:1]
        df_train = df_travel[1:50]
        df_test = df_travel[50:]
        df_data.reset_index(drop=True)

        mk_travel = MK_chain(df=df_data)
        model_acc = []

        for r_id in range(len(df_train)):
            mk_travel.fit(df_train.iloc[[r_id]])
            model_acc.append(evaluate_mk(df_test, mk_travel))

        epoch_acc.append(model_acc)
# %%
