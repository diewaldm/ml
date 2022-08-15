"""
Quelle: https://github.com/faizann24/Authorship-Attribution
"""

import os
import pandas as pd
import numpy as np

from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

DATA_FOLDER = "data/users"
ARTICLES_PER_AUTHOR = 200
AUTHORS_TO_KEEP = 6


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
    # classifier = SVC(kernel='linear')
    # classifier = GaussianNB()
    classifier = LogisticRegression()

    # Train classifier
    classifier.fit(trainData, trainLabels)

    # Get test predictions
    testPredictions = classifier.predict(testData)

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

print("Test Accuracy: %.2f\nTest Precision: %.2f\nTest Recall: %.2f\nTest Fscore: %.2f\n" % (
    np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(fscores)))

"""
show Confusion Matrix
"""
matrix = confusion_matrix(testLabels, testPredictions)
print(matrix)

ConfusionMatrixDisplay.from_estimator(classifier, testData, testLabels)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

disp = ConfusionMatrixDisplay(matrix, display_labels=ids[0:AUTHORS_TO_KEEP])
disp.plot(cmap=plt.cm.Blues)
plt.show()
