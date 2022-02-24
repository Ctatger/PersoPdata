import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

df_travel = pd.read_csv("csv/travel_based_dataframe.csv",index_col=0, converters={"start_gps_coord": literal_eval, "end_gps_coord": literal_eval, "travel_gps_list": literal_eval})

P_markov=[]
P_daybased_weekday=[]
P_daybased_weekend=[]
P_weekbased_weekday=[]
P_weekbased_weekend=[]


features_weekday=[P_markov,P_daybased_weekday,P_weekbased_weekday]
features_weekend=[P_markov,P_daybased_weekend,P_weekbased_weekend]
features_weekday=np.array(features_weekday)
features_weekend=np.array(features_weekend)

target=df_travel['gps_end_cluster']