# %%
# import matplotlib.pyplot as plt
from math import gamma
import seaborn as sns
import numpy as np

from data_parsing import create_dataframe


def Compute_Proba(data_frame, n_startcluster, n_endcluster, coeff_matrix=None):

    if coeff_matrix is None:
        coeff_matrix = np.ones((n_startcluster*n_endcluster, 2))*1e-6
        coeff_matrix = coeff_matrix.tolist()

    Start_clusters = data_frame['gps_start_cluster'].unique()
    End_clusters = data_frame['gps_end_cluster'].unique()

    for k in range(len(Start_clusters)):
        Total = 0
        Starting_points = data_frame.loc[data_frame['gps_start_cluster'] == Start_clusters[k]]
        for end_id in range(len(End_clusters)):
            Ending_points = Starting_points.loc[Starting_points['gps_end_cluster'] == End_clusters[end_id]]
            Total += len(Ending_points)

            if not coeff_matrix[k*len(End_clusters)+end_id]:
                coeff_matrix[k*len(End_clusters)+end_id].append(len(Starting_points))
                coeff_matrix[k*len(End_clusters)+end_id].append(len(Ending_points))
            else:
                coeff_matrix[k*len(End_clusters)+end_id][0] += len(Starting_points)
                coeff_matrix[k*len(End_clusters)+end_id][1] += len(Ending_points)

        if Total != len(Starting_points):
            raise ValueError("Probabilities not adding up to 1, check dataframe", Start_clusters[k])
    return coeff_matrix


def Create_ProbabilityMatrix(data_frame, coeff_mat=None):
    T_matrix = []
    State_prob = []
    prob = {}

    Start_clusters = data_frame['gps_start_cluster'].unique()
    End_clusters = data_frame['gps_end_cluster'].unique()

    for k in range(len(Start_clusters)):
        prob = {str(x): 0 for x in Start_clusters}

        for end_id in range(len(End_clusters)):
            State_prob.append([End_clusters[end_id],
                               coeff_mat[(k*len(End_clusters))+end_id][1]/coeff_mat[(k*len(End_clusters))+end_id][0]])

        for key in State_prob:

            prob[str(key[0])] = key[1]

        T_matrix.append(list(prob.values()))
    return T_matrix


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

    return gamma, TMat


def evaluate_mk(df, mk):

    acc_total = 0
    for _, row in df.iterrows():
        pred = mk.predict(str(row['gps_start_cluster']))
        answ = str(row['gps_end_cluster'])
        if pred == answ:
            acc_total += 1
    return (acc_total/len(df))*100


class MK_chain:
    def __init__(self, data_frame):
        """__init__ Markov chain Class constructor

        Arguments:
            Prob_mat  -- Transition matrix of the markov chain

        Raises:
            ValueError: Sum of probibilities in a given row should always add up to 1
        """

        self.trips_nb = []
        self.Start_clusters = []
        self.End_clusters = []
        self.sum_mat = []
        self.TOLERANCE_VALUE = 0.0001

        self.Start_clusters = data_frame['gps_start_cluster'].unique()
        self.Start_clusters.sort()
        self.End_clusters = data_frame['gps_end_cluster'].unique()
        self.End_clusters.sort()

        self.create_transition_matrix(data_frame)

        self.states = [str(x) for x in range(len(self.transitionMatrix))]

    def predict(self, curr_st, filter=False):
        if not filter:
            st_id = self.states.index(curr_st)
            pred_id = self.transitionMatrix[st_id].index(max(self.transitionMatrix[st_id]))
        else:
            st_id = self.states.index(curr_st)
            order = self.transitionMatrix[st_id].copy()
            order.sort(reverse=True)
            pred_id = self.transitionMatrix[st_id].index(order[1])
        return self.states[pred_id]

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

    def create_sum_matrix(self, data_frame):

        for Start in self.Start_clusters:
            Total = 0
            Starting_points = data_frame.loc[data_frame['gps_start_cluster'] == Start]

            for End in self.End_clusters:
                Ending_points = Starting_points.loc[Starting_points['gps_end_cluster'] == End]
                Total = len(Ending_points)

                self.sum_mat.append([Start, End, Total])

    def create_proba_vector(self, data_frame, Start):

        size = max(self.End_clusters[-1], self.Start_clusters[-1])
        proba_vector = np.zeros(size+1)
        starting_points = data_frame.loc[data_frame['gps_start_cluster'] == Start]
        sample = len(starting_points)

        for freq in self.sum_mat:
            if freq[0] == Start:
                proba_vector[freq[1]] = (freq[2]/sample)*100
        return proba_vector

    def create_transition_matrix(self, data_frame):

        self.create_sum_matrix(data_frame)

        size = max(self.End_clusters[-1], self.Start_clusters[-1])
        t_mat = np.array(self.create_proba_vector(data_frame, 0))

        for i in range(1, size+1):
            vect = self.create_proba_vector(data_frame, i)
            print(vect)
            t_mat = np.vstack((t_mat, vect))

        self.gamma = self.create_gamma(data_frame)
        self.transitionMatrix = t_mat

        return gamma, t_mat

    def create_gamma(self, df):
        gamma = []
        sample = len(df)

        for start in self.Start_clusters:
            df_s = df.loc[df['gps_start_cluster'] == start]
            gamma.append((len(df_s)/sample)*100)
        if sum(gamma) > 100.5:
            raise ValueError("Gamma matrix coefficients not adding up to 1")
        return gamma
# %%


if __name__ == "__main__":

    sns.set_theme()

    df_travel = create_dataframe()
    df_travel = df_travel.sample(frac=1)

    df_train = df_travel[:15]
    df_data = df_travel[15:]

    mk_travel = MK_chain(data_frame=df_train)

    """ J=0
    for i in range(5,len(df_travel),5):
        print(df_travel[J:i])
        J=i
    a = mk_travel.get_transmat() """

    test_df = df_travel.sample(frac=0.5)
    print(evaluate_mk(test_df, mk_travel))
# %%
