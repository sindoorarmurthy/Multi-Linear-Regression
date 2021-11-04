#Ravikumar Murthy, Sindoora
#1001862126
#Project 1 : Multivariate regression - Predicting Flower species
#COMPILATION STEP : python3 sindoora_MultiRegression.py

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sb 
import copy

flowerSpecies=['Iris-setosa', 'Iris-versicolor','Iris-virginica']
speciesValues=[1, 2, 3]
#read_data : function read the dataset from the folder and replace the Nominal values with some quantitative values.
def read_data():               
    data = pd.read_csv("./IRIS.csv")               #reads the dataset from the current working directory  
    sb.set(style="darkgrid", color_codes=True)
    g = sb.pairplot(data, hue="Species")     
    data.Species.replace(flowerSpecies,speciesValues , inplace=True)    #replacing the nominal values of column "species" in csv
    data = data.sample(frac=1).reset_index(drop=True)     #return all rows (in random order) and prevents reset_index from creating a column containing the old index entries.
    return data

# train_model : function to train the model. It returns betaCap for multivariate regression model.
def train_model(train_data):
    data = train_data.copy()
    train_data = data.drop('Species',axis = 1)        #drops species column for all rows
    train_data= train_data.values                                   
    #A -> matrix, Y -> 1,2,3
    #betaCap = (A-transpose.A)inv . (A-transpose.Y)
    betaCap = np.dot((inv(np.matmul(train_data.transpose(), train_data))),(np.matmul(train_data.transpose(),data.Species)))   
    return betaCap

#test_model : function is used to check the accuracy of our trained model. 
def test_model(betaCap,test_data):
    foo=[]  
    # selecting first 4 columns of test data with random indexs, multiplying betaCap with the test data cols to get the predicted values           
    predicted =(test_data.iloc[:,0:4].values*betaCap)  
    for i in range(len(predicted)):
        foo.append(round(sum(predicted[i])))    #rounding the predicted value to the nearest integer
    test_data.loc[:,'predicted']=foo            #creating new column "predicted" in test data
    test_data.loc[:,'accurate'] = np.where(test_data['Species']==test_data['predicted'], 1, 0)   #creating new column "accurate" in test data to mark whether the predicted value = actual value
    accuracy=test_data['accurate'].sum()/len(test_data)     #taking average of column "accurate" 
    return accuracy

#split_data : function to divide original dataframe into k parts. i.e., totalRowsInOriginalData/k = rows in each set
def split_data(lst,k):
    return [ lst[i::k] for i in range(k) ]

#kfold_CrossValidation : function to implement the k-fold cross validation. 
#It calls the train_model and test_model functions, finds the average of the accuracies and prints it to the user.
def kfold_CrossValidation(data,k):
    if k>1 and k<len(data):                               #checking if k is valid, else exit
        kDataSets = split_data(data, k)                    #dividing the dataset into k data sets    
        accuracyList = []
        for i in range(k):
            test_data = kDataSets[i].copy()               #assigning the first chunk of split data as test data
            train_data = data.drop(test_data.index)       #remaining rows will be assigned as train data
            betaCap = train_model(train_data)  
            accuracy = test_model(betaCap,test_data)
            accuracyList.append(accuracy)
        print('\nPerformance with k={} folds = {}%\n'.format(k,(np.mean(accuracyList)*100))); #calculate mean of accuracyList and convert to %
    else:
        print('k value is invalid')
        exit

#Main function , specify 'k' value
if __name__== "__main__":
  data = read_data()
  kfold_CrossValidation(data,3)  
  plot.show()
