import numpy as np
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.model_selection import KFold
import argparse
import csv
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC  
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
import seaborn as sns
import random

def plot_all(dia, sys, eda, res):
    
    
    rand_dia = np.random.choice(dia)
    rand_sys = np.random.choice(sys)
    rand_eda = np.random.choice(eda)
    rand_res = np.random.choice(res)

    plt.plot(dia, label='dia')
    plt.plot(sys, label='sys')
    plt.plot(eda, label='eda')
    plt.plot(res, label='res')
    
    plt.legend()
    plt.savefig('plot2.png')
    
    return 0
def plot_box(data, type):
    print(type)
    mean = type + ' mean'
    var = type + ' var'
    minimum = type + ' min'
    maximum = type + ' max'
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data, columns=[mean, var, minimum, maximum, 'Pain'])

    # Create a boxplot for each feature
    fig, axs = plt.subplots(ncols=4, figsize=(20,6))
    for i, col in enumerate([mean, var, minimum, maximum]):
        #print(col)
        boxprops = dict(linewidth=2, edgecolor='black', facecolor='white')
        flierprops = dict(marker='o', markerfacecolor='white', markersize=8,
                          linestyle='none', markeredgecolor='black')
        sns.boxplot(x='Pain', y=col, data=df, color='none', boxprops=boxprops,
                    flierprops=flierprops, ax=axs[i], width=0.1)
        axs[i].set_title(col)

        axs[i].set_title(col)

    title = type + "_boxplot2.png"
    plt.savefig(title)





# function to read and collect it in a list of the 4 features with the labels
def read_data(file, data):
    feature_vectors = []
    mean_list = []
    var_list = []
    min_list = []
    max_list = []
    org_list = []
    # open the csv file to read it
    with open(file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            sub_id = row[0]
            data_type = row[1]
            pain = row[2]
            # we check each row if it has the data as the the data type in the input 
            if data_type == data:
                # if it is we collect all the data 
                values_num = [float(x) for x in row[3:]]
                mean_list = np.mean(values_num) # get the mean of the collected row data
                var_list = np.var(values_num) # get the var of the collected row data
                min_list = np.min(values_num) # get the min of the collected row data
                max_list = np.max(values_num) # get the max of the collected row data
                
                # we add it each feature to the feature vectore and the append it it to the feature vectores 
                org_list = values_num
                feature_vector = [mean_list, var_list, min_list, max_list, pain]
                feature_vectors.append(feature_vector)
    #call pox plot function to get the plot of our data features pain and no pain for the spicific data type
    plot_box(feature_vectors, data)
    # we run the classifier function to train and test our data
    
    classifier(feature_vectors, data)
    return np.array(org_list).T

def classifier(feature_vectors, data_type):
    
    feature_vectors = np.array(feature_vectors)
    X = feature_vectors[:, :-1]
    y = feature_vectors[:, -1]
    #print(X, " ", y)
    
    # Define the K-fold cross-validation iterator
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # initialize the list that will store the accuracy, precision, recall, and confusion matrices
    avg_accuracy = []
    avg_precision = []
    avg_recall = []
    avg_conf_matrix = []

    # a loop to go through each fold
    for train_index, test_index in kf.split(X):
        # X_train, X_test, y_train, y_test are the training and testing sets created for each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_subjects = set(X_train[:, 0])
        test_subjects = set(X_test[:, 0])

        # check if there is any overlap
        if len(train_subjects.intersection(test_subjects)) != 0:
            raise ValueError("Some ERROR!")

        # using the SVM classifier
        clf = SVC(kernel='linear')
        clf.fit(X_train[:, 1:], y_train)

        y_pred = clf.predict(X_test[:, 1:])

        # calculate the accuracy, precision, recall, and the confusion matrix are calculated using the functions from the library
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="Pain")
        recall = recall_score(y_test, y_pred, pos_label="Pain")
        conf_matrix = confusion_matrix(y_test, y_pred, labels=["No Pain", "Pain"])

        avg_accuracy.append(accuracy)
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_conf_matrix.append(conf_matrix)

        
    # get the avg for all 
    avg_accuracy = np.mean(avg_accuracy)
    avg_precision = np.mean(avg_precision)
    avg_recall = np.mean(avg_recall)
    avg_conf_matrix = np.mean(avg_conf_matrix, axis=0)

    with open("project2.txt", "a") as f:
        f.write(f"{data_type}: \n")
        f.write(f"Average Accuracy: {avg_accuracy}\n")
        f.write(f"Average Precision: {avg_precision}\n")
        f.write(f"Average Recall: {avg_recall}\n")
        f.write(f"Average Confusion Matrix:\n{avg_conf_matrix}\n")
        f.write("============================\n")

    print(f"{data_type}:")
    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average Confusion Matrix:\n{avg_conf_matrix}")

    



def main():

    parser = argparse.ArgumentParser(description='Pain classification from cvs data')
    parser.add_argument('data_type', choices=['dia', 'sys', 'eda', 'res', 'all'], help='Type of data to use')
    parser.add_argument('data_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()

    print(args.data_type)
    print(args.data_file)

    # after getting the inputs from the command line
    # we check what kind of data type was entered to call the read data function with the spicific data type as it written in the cvs file
    if args.data_type == 'dia':
        data = 'BP Dia_mmHg'
        read_data(args.data_file, data)
    elif args.data_type == 'sys':
        data = 'LA Systolic BP_mmHg'
        read_data(args.data_file, data)
    elif args.data_type == 'eda':
        data = 'EDA_microsiemens'
        read_data(args.data_file, data)
    elif args.data_type == 'res':
        data = 'Respiration Rate_BPM'
        read_data(args.data_file, data)
    elif args.data_type == 'all':
        # if its all we change the data type 4 times for the 4 diffrent types and read the data and train it for each data type
        data = 'BP Dia_mmHg'
        dia = read_data(args.data_file, data)
        data = 'LA Systolic BP_mmHg'
        sys = read_data(args.data_file, data)
        data = 'EDA_microsiemens'
        eda = read_data(args.data_file, data)
        data = 'Respiration Rate_BPM'
        res = read_data(args.data_file, data)
        dia = np.array(dia)
        sys = np.array(sys)
        eda = np.array(eda)
        res = np.array(res)
        plot_all(dia, sys, eda, res)

    else:
        print("wrong input")

    

main()
