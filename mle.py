#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'] #column names

df = pd.read_csv('csv/processed.cleveland.data.csv', header = None)
df.columns = col_names # setting dataframe column names

# %%
    
df.replace({'?': np.nan}, inplace = True) # converting '?' to NaN values
df[['ca', 'thal']] = df[['ca', 'thal']].astype('float64') # Casting columns data-type to floats
df['ca'].replace({np.nan: df['ca'].median()}, inplace = True) # replaces null values of ca column with median value
df['thal'].replace({np.nan: df['thal'].median()}, inplace = True)

# %%
# selecting all the features within our dataset
features = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']] 
features = features.to_numpy() # converts feature set to numpy array
target = df['num'].to_numpy() # converts target column to numpy array
features.shape, len(target) 
# %%
# function for standardizing data
def standardScaler(feature_array):
    """Takes the numpy.ndarray object containing the features and performs standardization on the matrix.
    The function iterates through each column and performs scaling on them individually.
    
    Args-
        feature_array- Numpy array containing training features
    
    Returns-
        None
    """
    
    total_cols = feature_array.shape[1] # total number of columns 
    for i in range(total_cols): # iterating through each column
        feature_col = feature_array[:, i]
        mean = feature_col.mean() # mean stores mean value for the column
        std = feature_col.std() # std stores standard deviation value for the column
        feature_array[:, i] = (feature_array[:, i] - mean) / std # standard scaling of each element of the column
# %%
standardScaler(features) # performing standardization on our feature set 

# checking if standardization worked
total_cols = features.shape[1] # total number of columns 
for i in range(total_cols):
    print(features[:, i].std())
# %%
# creating randomized weights for our linear predictor func
weights = np.random.rand(5, 13)
# creating randomized biases for our linear predictor func
biases = np.random.rand(5, 1)

def linearPredict(featureMat, weights, biases):
    """This is the linear predictor function for out MLR model. It calculates the logit scores for each possible outcome.
    
    Args-
        featureMat- A numpy array of features
        weights- A numpy array of weights for our model
        biases- A numpy array of biases for our model
    
    Returns-
        logitScores- Logit scores for each possible outcome of the target variable for each feature set in the feature matrix
    """
    logitScores = np.array([np.empty([5]) for i in range(featureMat.shape[0])]) # creating empty(garbage value) array for each feature set
    
    for i in range(featureMat.shape[0]): # iterating through each feature set
        logitScores[i] = (weights.dot(featureMat[i].reshape(-1,1)) + biases).reshape(-1) # calculates logit score for each feature set then flattens the logit vector 
    
    return logitScores

# %%
def softmaxNormalizer(logitMatrix):
    """Converts logit scores for each possible outcome to probability values.
    
    Args-
        logitMatrix - This is the output of our logitPredict function; consists  logit scores for each feature set
    
    Returns-
        probabilities - Probability value of each outcome for each feature set
    """
    
    probabilities = np.array([np.empty([5]) for i in range(logitMatrix.shape[0])]) # creating empty(garbage value) array for each feature set

    for i in range(logitMatrix.shape[0]):
        exp = np.exp(logitMatrix[i]) # exponentiates each element of the logit array
        sumOfArr = np.sum(exp) # adds up all the values in the exponentiated array
        probabilities[i] = exp/sumOfArr # logit scores to probability values
    return probabilities

def multinomialLogReg(features, weights, biases):
    """Performs logistic regression on a given feature set.
    
    Args- 
        features- Numpy array of features(standardized)
        weights- A numpy array of weights for our model
        biases- A numpy array of biases for our model
    
    Returns-
        probabilities, predictions
        Here,
            probabilities: Probability values for each possible outcome for each feature set in the feature matrix
            predictions: Outcome with max probability for each feature set
    """
    logitScores = linearPredict(features, weights, biases)
    probabilities = softmaxNormalizer(logitScores)
    predictions = np.array([np.argmax(i) for i in probabilities]) #returns the outcome with max probability
    return probabilities, predictions

def accuracy(predictions, target):
    """Calculates total accuracy for our model.
    
    Args- 
        predictions- Predicted target outcomes as predicted by our MLR function
        target- Actual target values
    
    Returns-
        accuracy- Accuracy percentage of our model
    """
    correctPred = 0
    for i in range(len(predictions)):
        if predictions[i] == target[i]:
            correctPred += 1
    accuracy = correctPred/len(predictions)*100
    return accuracy

#%%
def train_test_split(dataframe, test_size = 0.2):
    """Splits dataset into training and testing sets.
    
    Args- 
        dataframe- The dataframe object you want to split
        test_size- Size of test dataset that you want
    
    Returns-
        train_features, train_target, test_features, test_target 
    """
    
    data = dataframe.to_numpy() # converts dataframe to numpy array
    totalRows = data.shape[0] # total rows in the dataset
    testRows = np.round(totalRows * test_size) # total rows in testing dataset
    randRowNum = np.random.randint(0, int(totalRows), int(testRows)) # randomly generated row numbers
    testData = np.array([data[i] for i in randRowNum]) # creates test dataset
    data = np.delete(data, randRowNum, axis = 0) # deletes test data rows from main dataset; making it training dataset
    train_features = data[:, :-1]
    train_target = data[:, -1]
    test_features = testData[:, :-1]
    test_target = testData[:, -1]
    
    return train_features, train_target, test_features, test_target    

# running train_test_split for our dataset
train_features, train_target, test_features, test_target = train_test_split(df, test_size = 0.17)
standardScaler(train_features) # standard scaling training set 
standardScaler(test_features) # standard scaling testing set
#%%
def crossEntropyLoss(probabilities, target):
    """Calculates cross entropy loss for a set of predictions and actual targets.
    
    Args-
        predictions- Probability predictions, as returned by multinomialLogReg function
        target- Actual target values
    Returns- 
        CELoss- Average cross entropy loss
    """
    n_samples = probabilities.shape[0]
    CELoss = 0
    for sample, i in zip(probabilities, target):
        CELoss += -np.log(sample[i])
    CELoss /= n_samples
    return CELoss   

def stochGradDes(learning_rate, epochs, target, features, weights, biases):
    """Performs stochastic gradient descent optimization on the model.
    
    Args-
        learning_rate- Size of the step the function will take during optimization
        epochs- No. of iterations the function will run for on the model
        target- Numpy array containing actual target values
        features- Numpy array of independent variables
        weights- Numpy array containing weights associated with each feature
        biases- Array containinig model biases
    
    Returns-
        weights, biases, loss_list
        where,
            weights- Latest weight calculated (Numpy array)
            bias- Latest bias calculated (Numpy array)
            loss_list- Array containing list of losses observed after each epoch    
    """
    target = target.astype(int)
    loss_list = np.array([]) #initiating an empty array
    
    for i in range(epochs):
        probabilities, _ = multinomialLogReg(features, weights, biases) # Calculates probabilities for each possible outcome
        
        CELoss = crossEntropyLoss(probabilities, target) # Calculates cross entropy loss for actual target and predictions
        loss_list = np.append(loss_list, CELoss) # Adds the CELoss value for the epoch to loss_list
        
        probabilities[np.arange(features.shape[0]),target] -= 1 # Substract 1 from the scores of the correct outcome
        
        grad_weight = probabilities.T.dot(features) # gradient of loss w.r.t. weights
        grad_biases = np.sum(probabilities, axis = 0).reshape(-1,1) # gradient of loss w.r.t. biases
        
        #updating weights and biases
        weights -= (learning_rate * grad_weight)
        biases -= (learning_rate * grad_biases)
        
    return weights, biases, loss_list

updatedWeights, updatedBiases, loss_list = stochGradDes(0.1, 2000, train_target, train_features, weights, biases)

testProbabilities, testPredictions = multinomialLogReg(test_features, updatedWeights, updatedBiases)

correctPreds = 0
for i in range(len(testPredictions)):
    if testPredictions[i] == test_target[i]:
        correctPreds += 1
acc = correctPreds / len(testPredictions) * 100
print("Model accuracy on test dataset - {}".format(acc))
#%%
# %%
