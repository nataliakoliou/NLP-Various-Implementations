"""
AG News Topic Classification: Building on the scikit-learn library of machine learning functions, implement them following text classification approaches (in all cases, the texts to are converted to lowercase):
• Multinomial Naïve Bayes3 with text representation according to the approach
tf-idf word uni-grams4.
"""

import csv
import time
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = list(csv.reader(f))
    return {"features": [row[1] + ' ' + row[2] for row in data[1:]], "labels": [row[0] for row in data[1:]]}

def extract_features(train_text, test_text, n, approach):
    vectorizer = TfidfVectorizer(ngram_range=(n,n), lowercase=True, analyzer=approach) # where (n,n): the lower and upper bounds of the range of n-grams to be extracted
    return vectorizer.fit_transform(train_text), vectorizer.transform(test_text), vectorizer.vocabulary_

def train_classifier(X_train, y_train, classifier):
    return classifier.fit(X_train, y_train)

def predict(classifier, X_test):
    return classifier.predict(X_test)

def run_model(train_data, test_data, n, approach, classifier):
    start = time.time()
    X_train, X_test, vocab = extract_features(train_data["features"], test_data["features"], n, approach) # (i, j) entries represent the presence and frequency of the j-th feature (word) in the i-th document - the values in each entry represent the corresponding term frequency-inverse document frequency (tf-idf) score
    y_train, y_test = train_data["labels"], test_data["labels"]
    clf = train_classifier(X_train, y_train, classifier)
    end = time.time()
    return clf.score(X_test, y_test), len(vocab), end - start

def visualize(model, accuracy, dimensionality, time_cost):
    print(f"\033[1m{model}:\033[0m")
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Accuracy", "Dimensionality", "Time Cost"]])
    pt.add_row([accuracy, dimensionality, time_cost])
    print(pt)

train_data = load_dataset("C:/Users/natalia/pyproj/nlp-proj/assignment-2b/train.csv")
test_data = load_dataset("C:/Users/natalia/pyproj/nlp-proj/assignment-2b/test.csv") 

accuracy, dimensionality, time_cost = run_model(train_data, test_data, 1, "word", MultinomialNB())
visualize("Multinomial Naïve Bayes with tf-idf word uni-grams", accuracy, dimensionality, time_cost)

accuracy, dimensionality, time_cost = run_model(train_data, test_data, 3, "char", MultinomialNB())
visualize("Multinomial Naïve Bayes with tf-idf character tri-grams", accuracy, dimensionality, time_cost)
