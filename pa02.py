# EE532L - Deep Learning for Healthcare - Programming Assignment 02
# Important: Please do not change/rename the existing function names and write your code only in the place where you are asked to do it.
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Write your code for the PCA below 
def my_PCA(X):
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    
    # covariance matrix
    cov_matrix = np.cov(X_centered)
    
    # Eigen values and vectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort the eigenvectors by decreasing eigenvalues
    index = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, index]
    
    # Project the data onto the principal components
    data = np.dot(eigenvectors.T, X_centered)

    
    return data




def regress_fit(X_train, y_train, X_test):
    
    X_train = np.array(X_train) # Normalizing training features
    c = np.expand_dims(np.amax(X_train, axis=1),axis=1)
    X_train = X_train / c
    
    X_test = np.array(X_test) # Normalizing testing features
    c = np.expand_dims(np.amax(X_test, axis=1),axis=1)
    X_test = X_test / c
    
    p = X_train.shape[0] # Number of features
    N = X_train.shape[1] # Number of sample cases
    e = 0.0001 # Learning Rate (Hint:- adjust between 1e-5 to 1e-2)
    
    w = np.random.uniform(-1/np.sqrt(p), 1/np.sqrt(p), (p+1,1)) # Random initialization of weights
    X = np.ones((p+1,N)) # Adding an extra column of ones to adjust biases
    X[:p,:] = X_train
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    ious = []

    TP,TN,FP,FN = 0,0,0,0
    num_epochs = 3000
    for epoch in range(num_epochs): # Loop for iterative process
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        for i in range (N):
            z = ((w.T)@X[:,i:i+1])[0,0] # Raw logits (W.T x X)    
            
            
            y = 1/(1 + np.exp(-z)) # Sigmoid activation function
            
            T = y_train[i] # Ground Truth
           
           
            J = J-(T * np.log(y) + (1 - T) * np.log(1 - y)) # Loss function    
           
            
           
            k = (((1-T)/(1-y))-(T/y))*y*(1-y) # Derivative of J w.r.t z (Chain rule, J w.r.t y multiplied by y w.r.t z )
            dJ = k*X[:,i:i+1] # Final Derivative of J w.r.t w (dJ/dz multiplied by dz/dw)
            
           
            w = w - e * dJ # Gradient Descent
            
            if abs(y-T)<0.5:
                count = count+1 # Counting the number of correct predictions
                
            
            if y >= 0.5 and T == 1:
                TP += 1
            elif y < 0.5 and T == 0:
                TN += 1
            elif y >= 0.5 and T == 0:
                FP += 1
            elif y < 0.5 and T == 1:
                FN += 1

        train_loss = J/N
        train_accuracy = 100*count/N
        losses.append(train_loss)
        accuracies.append(train_accuracy)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precisions.append(precision)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(specificity)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        ious.append(iou)

    batch_metrics = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} \n"
    sys.stdout.write('\r' + batch_metrics)
    sys.stdout.flush()

  
    epochs = np.arange(1, num_epochs+1)
    print(np.shape(epochs))
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')

    plt.subplot(2, 3, 2)
    plt.plot(epochs, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')

    plt.subplot(2, 3, 3)
    plt.plot(epochs, precisions)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision vs Epoch')

    plt.subplot(2, 3, 4)
    plt.plot(epochs, recalls)
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall vs Epoch')

    plt.subplot(2, 3, 5)
    plt.plot(epochs, specificities)
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epoch')

    plt.subplot(2, 3, 6)
    plt.plot(epochs, f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')

    plt.tight_layout()
    plt.show()

    # Testing
         
    N2 = X_test.shape[1] # Number of test samples

    X2 = np.ones((p+1,N2)) # adding an additional columns of 1 to adjust biases
    X2[:p,:] = X_test

    z2 = w.T@X2 # test logit matrix
    y_pred = 1/(1+np.exp(-z2)) # Sigmoid activation function to convert into probabilities
    y_pred[y_pred>=0.5] = 1 # Thresholding
    y_pred[y_pred<0.5] = 0
    return y_pred

def load_and_fit():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
   

    # Combine features and target variable for shuffling
    data = np.hstack((X, np.array(y).reshape(-1, 1)))

    np.random.seed(208)
    np.random.shuffle(data)

    X2 = data[:, :-1]
    y = data[:, -1]

    X2 = X2.T
    X2 = my_PCA(X2)

    X_train = X2[:, :614]
    X_test = X2[:, 614:]
    y_train = y[:614]
    y_test = y[614:]

   
    
    # Fit the model
    y_test_pred = regress_fit(X_train, y_train, X_test)

    # Evaluate the accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred[0])
    print(f"Test Accuracy: {test_accuracy:.5f}")
    return round(test_accuracy, 5)

a = load_and_fit()
