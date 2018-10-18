#!/usr/bin/env python3 -w ignore DataConversionWarning
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def boxplot(column):
    # to do: return a boxplot of each variables
    return 0

def plotHist(column, title, x_label, y_label):
    # to do: plot histogram for each individual variable
    binwidth = [x for x in range(0,20000, 2000)]
    ex = plt.hist(column, bins=binwidth)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return plt.show()

def plotHistTwo(colA, colB, title="", x_label="", y_label="frequency"):
    # to do: plot a two way histogram for male female for each variable
    binwidth = [x for x in range(0,30000, 1000)]
    # plt.hist(colA, bins=binwidth, alpha=0.5, label = "favNumberMales")
    # plt.hist(colB, bins=binwidth, alpha=0.5, label = "favNumberFemales")
    plt.hist([colA, colB], bins=binwidth, alpha=0.5, label=["tweetCountMales", "tweetCountFemales"])
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return plt.show()

def scatter(col1, col2):
    # to do: plot a scatter plot for variables. E.g. hue vs brightness with
    # male and female colored differently
    return 0

def main():
    # start time
    startTime = time.time()

    # load the dataset
    dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/dataset/overall_dataset.csv'
    data = pd.read_csv(dataset, na_values = '?')

    # change appropriate variables to categorical. DON'T DO THIS!
    # data['gender'] = data['gender'].astype('category')
    # data['tweet_location'] = data['tweet_location'].astype('category')
    # data['user_timezone'] = data['user_timezone'].astype('float64')

    # reformat date column
    data['created'] = pd.to_datetime(data['created'])

    # create new columns for year and month
    data['year'] = pd.DatetimeIndex(data['created']).year
    data['month'] = pd.DatetimeIndex(data['created']).month

    # remove original date column
    data = data.drop(['created'], axis=1)

    # standardize numeric variables (could also consider using robust scaler here)
    numericVariables = ['fav_number', 'tweet_count','retweet_count', 'link_hue',
     'link_sat', 'link_vue', 'sidebar_hue', 'sidebar_sat', 'sidebar_vue', 'year', 'month']
    scaler = preprocessing.StandardScaler()
    data[numericVariables] = scaler.fit_transform(data[numericVariables])

    # create dependent & independent variables
    X = data.drop('gender', axis=1)
    y = data['gender']

    # split into 90% training, 10% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    # train model
    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(X_train, y_train)

    # make predictions
    y_pred = svcclassifier.predict(X_test)

    # print metrics
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))


#     # create a subset of males and females
# #     males = data[data['gender']==0]
# #     females = data[data['gender']==1]
# #
# #     # to access specific columns
# #     favNumberMales = males.loc[:,'fav_number']
# #     favNumberFemales = females.loc[:,'fav_number']
# # #    plotHistTwo(favNumberMales, favNumberFemales)
# #
# #     tweetCountMales = males.loc[:,'tweet_count']
# #     tweetCountFemales = females.loc[:,'tweet_count']
#     # plotHistTwo(tweetCountMales, tweetCountFemales)
#
#     # retweetCountMales = males.loc[:,'retweet_count']
#     # retweetCountFemales = females.loc[:,'retweet_count']
#
#     # plot a histogram
#     #plot_hist(fav_number, "title", "favourited tweets", "freq")
#

    # to keep track of time taken
    endTIme = time.time()
    totalTime = endTIme - startTime
    print("Time taken:", totalTime)

if __name__ == '__main__':
  main()

#
# see notes on repl
# to do: compute decision tree with chosen dependent variables
# return: recall precision and f1 score for decision tree
#
# to do: compute logisitic regression
# return: recall precision and f1
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html.
#
# to do: compute SVM
# return: recall precision and f1
