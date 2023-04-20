import csv
import time
import random
from prettytable import PrettyTable
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

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
    misclass_data = detect_misclassification(test_data, clf.predict(X_test))
    return clf.score(X_test, y_test), len(vocab), end - start, misclass_data

def visualize(model, accuracy, dimensionality, time_cost):
    print(f"\033[1m{model}:\033[0m")
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Accuracy", "Dimensionality", "Time Cost"]])
    pt.add_row([accuracy, dimensionality, time_cost])
    print(pt)

def detect_misclassification(test_data, y_pred):
    misclass_data = defaultdict(list)
    for i in range(len(test_data["labels"])):
        true_label = test_data["labels"][i]
        predicted_label = y_pred[i]
        if true_label != predicted_label:
            text = test_data["features"][i]
            misclass_data[true_label].append((text, predicted_label))
    return misclass_data

def analyze_results(mnv1w, mnv3c, svm1w, svm3c):
    
    common_misclass_data = defaultdict(list)
    for true_label in mnv1w.keys():
        for text, label in mnv1w[true_label]:
            labels = [label] + [next((p for t, p in model[true_label] if t == text), '') for model in [mnv3c, svm1w, svm3c]]
            common_misclass_data[true_label].append((text, labels)) if all(labels) else None
    
    misclass_counts = {true_label: len(misclass_tuple) for true_label, misclass_tuple in common_misclass_data.items()}
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["True Label", "Misclassification Times"]])
    pt.add_rows([(true_label, count) for true_label, count in misclass_counts.items()])
    print(pt)

    misclass_freqs = defaultdict(int)
    for true_label, values in common_misclass_data.items():
        for text, pred_labels in values:
            for pl in pred_labels:
                misclass_freqs[(true_label, pl)] += 1
    max_tuple, max_count = max(misclass_freqs.items(), key=lambda x: x[1])
    print("\033[1m" + "Most common Misclassification Pair:" + "\033[0m")
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["True Label", "Predicted Label", "Frequency"]])
    pt.add_row([max_tuple[0], max_tuple[1], max_count])
    print(pt)

    rand_true_label = random.choice(list(common_misclass_data.keys()))
    rand_misclass_tuple = random.choice(common_misclass_data[rand_true_label])
    print("\033[1m" + "Text: " + "\033[0m" + rand_misclass_tuple[0] + "\033[1m" + "\nTrue Label: " + "\033[0m" + rand_true_label)
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Model", "Prediction"]])
    pt.add_rows([(model, rand_misclass_tuple[1][idx]) for idx, model in enumerate(["mnv1w", "mnv3c", "svm1w", "svm3c"])])
    print(pt)

train_data = load_dataset("C:/Users/natalia/pyproj/nlp-proj/assignment-2b/train.csv")
test_data = load_dataset("C:/Users/natalia/pyproj/nlp-proj/assignment-2b/test.csv") 

accuracy, dimensionality, time_cost, misclass_data_mnv1w = run_model(train_data, test_data, 1, "word", MultinomialNB())
visualize("Multinomial Naïve Bayes using tf-idf word uni-grams", accuracy, dimensionality, time_cost)

accuracy, dimensionality, time_cost, misclass_data_mnv3c = run_model(train_data, test_data, 3, "char", MultinomialNB())
visualize("Multinomial Naïve Bayes using tf-idf character tri-grams", accuracy, dimensionality, time_cost)

accuracy, dimensionality, time_cost, misclass_data_svm1w = run_model(train_data, test_data, 1, "word", LinearSVC(C=1))
visualize("Support Vector Machines using tf-idf word uni-grams", accuracy, dimensionality, time_cost)

accuracy, dimensionality, time_cost, misclass_data_svm3c = run_model(train_data, test_data, 3, "char", LinearSVC(C=1))
visualize("Support Vector Machines using tf-idf character tri-grams", accuracy, dimensionality, time_cost)

analyze_results(misclass_data_mnv1w, misclass_data_mnv3c, misclass_data_svm1w, misclass_data_svm3c)
