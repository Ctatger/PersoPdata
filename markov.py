from ast import Raise
from operator import index
import numpy as np
import random as rm



class MK_chain :
    
    def __init__(self,Prob_mat) -> None:
        self.TOLERANCE_VALUE=0.0001
        self.transitionMatrix=Prob_mat
        self.states=[str(x-1) for x in range(len(self.transitionMatrix))]

        for i in range(len(self.states)):
            if abs(1 - sum(self.transitionMatrix[i])) > self.TOLERANCE_VALUE:
                raise ValueError("probabilities don't add up to 1 (please keep the 10e-3 threshold in mind)",self.transitionMatrix[i])

    def predict(self,curr_st,filter=False):
        if not filter:
            st_id=self.states.index(curr_st)
            pred_id=self.transitionMatrix[st_id].index(max(self.transitionMatrix[st_id]))
        else:
            st_id=self.states.index(curr_st)
            order=self.transitionMatrix[st_id].copy()
            order.sort(reverse=True)
            pred_id=self.transitionMatrix[st_id].index(order[1])
        return self.states[pred_id]

    def update_transmat(self,st,prob_list):
        if abs(1-sum(prob_list)) > self.TOLERANCE_VALUE:
            raise ValueError("New probabilities don't add up to 1 (please keep the 10e-3 threshold in mind)")

        state_id=self.states.index(st)
        self.transitionMatrix[state_id]=prob_list
    
    def add_state(self,new_state,new_probs,old_probs):

        for k in old_probs:
            if abs(1-sum(k)) > self.TOLERANCE_VALUE:
                raise ValueError("New probabilities for existing states don't add up to 1 (please keep the 10e-3 threshold in mind)")

        if abs(1-sum(new_probs)) > self.TOLERANCE_VALUE:
            raise ValueError("Probabilities for new state don't add up to 1 (please keep the 10e-3 threshold in mind)")

        for i in range(len(self.states)):
            self.transitionMatrix[i]=old_probs[i]
        
        self.states.append(new_state)
        self.transitionMatrix.append(new_probs)
    
    def get_transmat(self):
        return self.transitionMatrix

    def get_states(self):
        return self.states

if __name__ == "__main__":

    transitionMatrix = [[0.2,0.6,0.2],[0.1,0.3,0.6],[0.2,0.7,0.1]]

    Mk=MK_chain(transitionMatrix)

    print(Mk.get_states())
    print(Mk.get_transmat())
    print(Mk.predict("0"))