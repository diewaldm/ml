import os
import pandas as pd
import io
import sys
import argparse
import numpy as np

from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

DATA_FOLDER = "data/users"
ARTICLES_PER_AUTHOR = 250
AUTHORS_TO_KEEP = 6


def calculate_top5accuracy(calc_labels, prediction_probs):
    """
    Takes as input labels and prediction probabilities and calculates the top-5 accuracy of the model
    """
    acc = []
    for j in range(0, len(prediction_probs)):
        pre_probs = prediction_probs[j]
        pre_probs_indices = np.argsort(-pre_probs)[:5]
        if calc_labels[j] in pre_probs_indices:
            acc.append(1)
        else:
            acc.append(0)

    return round(((acc.count(1) * 100) / len(acc)), 2)


print("Data loading...")
author_array = []
ids = []
for root, dirs, files in os.walk(DATA_FOLDER):
    for name in files:
        author_array.append(pd.read_csv(DATA_FOLDER + '/' + name))
        ids.append(name.split('_')[0])

authorArticles = []
labels = []
authorId = 0

for author in author_array:
    for x in range(ARTICLES_PER_AUTHOR):
        authorArticles.append(author.loc[x, "Beitrag"])
        labels.append(authorId)
    authorId = authorId + 1
    if authorId == AUTHORS_TO_KEEP:
        break

print("\nTraining and testing...")
# Train and get results
accuracies, precisions, recalls, fscores, top5accuracies = [], [], [], [], []
for i in range(5):
    # Train and test 5 different times and average the results

    # Split data into training and testing
    trainData, testData, trainLabels, testLabels = train_test_split(authorArticles, labels, test_size=0.2)

    # Convert raw corpus into tfidf scores
    vectorizer = TfidfVectorizer(min_df=15)
    vectorizer.fit(trainData)
    trainData = vectorizer.transform(trainData).toarray()
    testData = vectorizer.transform(testData).toarray()

    # Create a classifier instance
    classifier = RandomForestClassifier(n_estimators=100)

    # Train classifier
    classifier.fit(trainData, trainLabels)

    # Get test predictions
    testPredictions = classifier.predict(testData)
    testPredictionsProbs = classifier.predict_proba(testData)
    testTopFiveAccuracy = calculate_top5accuracy(testLabels, testPredictionsProbs)

    # Calculate metrics
    accuracy = round(accuracy_score(testLabels, testPredictions) * 100, 2)
    precision = round(precision_score(testLabels, testPredictions, average='macro') * 100, 2)
    recall = round(recall_score(testLabels, testPredictions, average='macro') * 100, 2)
    fscore = round(f1_score(testLabels, testPredictions, average='macro') * 100, 2)
    confusionMatrix = confusion_matrix(testLabels, testPredictions)

    # Store metrics in lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)
    top5accuracies.append(testTopFiveAccuracy)

print("Top-5 Test Accuracy: %.2f\nTest Accuracy: %.2f\nTest Precision: %.2f\nTest Recall: %.2f\nTest Fscore: %.2f\n" % (
    np.mean(top5accuracies), np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(fscores)))

"""
show Confusion Matrix
"""
# matrix = confusion_matrix(testLabels, testPredictions)
# print(matrix)
#
# ConfusionMatrixDisplay.from_estimator(classifier, testData, testLabels)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
#
# disp = ConfusionMatrixDisplay(matrix, display_labels=ids[0:AUTHORS_TO_KEEP])
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
