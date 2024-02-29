# EE532L - Deep Learning for Healthcare - Programming Assignment 02
# Important: Please do not change/rename the existing function names and write your code only in the place where you are asked to do it.
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import numpy as np

# Write your code for the PCA below 
def my_PCA(X):


    return(data)

def regress_fit(X_train, y_train, X_test):
    
    X_train = np.array(X_train) # Normalizing training features
    c = np.expand_dims(np.amax(X_train, axis = 1),axis=1)
    X_train = X_train / c
    
   
    
    X_test = np.array(X_test) # Normalizing testing features
    c = np.expand_dims(np.amax(X_test, axis = 1),axis=1)
    X_test = X_test / c
    
    p = X_train.shape[0] # Number of features
    N = X_train.shape[1] # Number of sample cases
    e = 0.000001 # Learning Rate (Hint:- adjust between 1e-5 to 1e-2)
    
    w = np.random.uniform(-1/np.sqrt(p), 1/np.sqrt(p), (p+1,1)) # Random initialization of weights
    X = np.ones((p+1,N)) # Adding an extra column of ones to adjust biases
    X[:p,:] = X_train
   
    
    # eps = 0.000000001 # infinitesimal element to avoid nan
    num_epochs = 2000
    for epoch in range(num_epochs): # Loop for iterative process
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        for i in range (N):

        


    return y_pred
###########################################################################################################################################


########################################################## Cannot be modified ##############################################################
# Load the dataset
def load_and_fit():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)

    X2 = np.array(X)
    np.random.seed(208)
    np.random.shuffle(X2)
    
    X2 = X2.T
    X2 = my_PCA(X2)
    y = df["Outcome"]
    X_train = X2[:,:614]
    
    X_test = X2[:,614:]
    y_train = y[:614]
    
    y_test = y[614:]
    
    

    # Fit the model
    y_test_pred = regress_fit(X_train, y_train, X_test)

    # Evaluate the accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred[0])
    print(f"Test Accuracy: {test_accuracy:.5f}")
    return round(test_accuracy, 5)

a = load_and_fit()
