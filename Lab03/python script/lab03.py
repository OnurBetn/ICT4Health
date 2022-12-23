import pandas as pd
import numpy as np
from sklearn import tree
from graphviz import render


def RidgeRegr(X, y, lamb):
    Nf = X.shape[1]     
    I = np.identity(Nf)
    w = np.linalg.inv(np.dot(X.T, X) + lamb * I).dot(X.T).dot(y)    
    return w


if __name__ == '__main__':

    # Read the attributes' names from the first rows of the file
    columns = pd.read_csv('chronic_kidney_disease.arff', nrows=25, skiprows=2, sep='\s+', header=None)[1]

    # Read data
    data = pd.read_csv('chronic_kidney_disease.arff',
                       sep='\s*,+\s*',  # regex to choose as separator one or more commas followed and preceded by zero or more whitespaces
                       skiprows=29,
                       names=columns, usecols=columns,
                       engine='python',
                       na_values='?')
    
    # Remove patients with less than 20 valid data
    data.dropna(thresh=21, inplace=True)
    
    # Map categorical features into numbers
    data_num = data.replace(['yes','good','present','normal','ckd'], 1)
    data_num = data_num.replace(['no','poor','notpresent','abnormal','notckd'], 0)
    
    # Select patients with 25 valid data
    X = data_num.dropna()
    
    # Normalizing data
    X_norm = (X - X.mean()) / X.std()
    data_mean = data_num.mean()
    data_std = data_num.std()
    data_norm = (data_num - data_mean) / data_std
    
    # Find a matrix W which has the weight vector of the j-th feature in the j-th column
    Nf = data.shape[1]  # Number of features
    W = np.zeros([Nf-1, Nf-1])  # Matrix of weights
    for j in range(Nf-1):
        data_train = X_norm.drop(X_norm.columns[j], axis=1).values
        y_train = X_norm.values[:, j]
        W[:, j] = RidgeRegr(data_train, y_train, lamb=10)
    
    nan_entries = np.argwhere(pd.isna(data_norm.values))  # Indices of the NaN values
    data_norm.replace(np.nan, 0, inplace=True)  # Replace NaNs into 0s to not keep them into consideration while regressing
    data_reg = data_num.copy()  # Final data in which regressed values will be put
    
    # Perform the regression for each of the NaNs entries
    for ix in nan_entries:
        r = ix[0]
        c = ix[1]
        x = np.delete(data_norm.values[r, :], c)
        y_hat = x.dot(W[:, c]) * data_std.iloc[c] + data_mean.iloc[c]
        
        # Guarantee that every regressed feature has a valid value
        if c == 2:  # Feature 'sg'
            possible_values = [1.005, 1.010, 1.015, 1.020, 1.025]
            data_reg.iloc[r, c] = min(possible_values, key=lambda x:abs(x-y_hat))
        elif c in [3, 4]:  # Features 'al' and 'su' in range [0-5]
            possible_values = range(6)
            data_reg.iloc[r, c] = min(possible_values, key=lambda x:abs(x-y_hat))
        elif c in list(range(5, 9)) + list(range(18, 25)):  # Categorical features mapped as 0/1
            possible_values = [0, 1]
            data_reg.iloc[r, c] = min(possible_values, key=lambda x:abs(x-y_hat))
        elif c in [11, 13, 14, 17]:  # Features to be round with one decimal digit
            data_reg.iloc[r, c] = round(y_hat, 1)
        else:  # Features to be round to integers
            data_reg.iloc[r, c] = round(y_hat)

    # Generate the decision tree
    clf = tree.DecisionTreeClassifier("entropy")
    clf.fit(data_reg.iloc[:, 0:24], data_reg.iloc[:, 24])

    # Print the features' importance
    print("Features' importance:")
    for i in range(Nf-1):
        print('\t', columns[i], clf.feature_importances_[i])
    
    # Create the .dot file of the decision tree
    tree.export_graphviz(clf, out_file="Tree.dot",
                         feature_names=columns.iloc[:24],
                         class_names=['not ckd', 'ckd'],
                         filled=True,
                         rounded=True,
                         special_characters=True)
    
    # Create the .png image of the decision tree from the .dot file
    render('dot', 'png', 'Tree.dot')
