# NLP-Various-Implementations
This repository contains all NLP course assignments.

## Assignment.1a | Tokenization using NLTK, SpaCy, and BERT
This project provides a Python implementation of tokenization and frequency analysis on text data using three different libraries: NLTK, SpaCy, and Transformers. It includes functions to tokenize text using each of these libraries and then analyze the frequency of the resulting tokens, providing tables and graphs to visualize the results.

* **Implement manual feature extraction:** Identifies and describes the most common features that define the internal structure of the text-datasets (training & testing). These features include: Single Letter Frequencies, Letter Occurencies in k-letter words, Letter Position Frequencies and Double Letters Frequencies.
* **Perform manual feature selection:** Creates feature-set (X) and label-set (y), by selecting the features that describe best each class.
* **Implement the classification model iteratively:** Trains an SVM classifier on the training plaintext. It then uses this classification model iteratively, to assign class-labels to the testing ciphertext (decryption alphabet prediction).
* **Decrypt the testing ciphertext:** Applies the predicted decryption alphabet to the testing ciphertext to decrypt it.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* NumPy: ```pip install numpy```
* Scikit-learn: ```pip install -U scikit-learn```
* Matplotlib: ```pip install matplotlib```
* NLTK: ```pip install nltk```
* Spacy: ```pip install spacy```
* Transformers: ```pip install transformers```
* PrettyTable: ```pip install prettytable```

**Alternatively:** you can download [requirements.txt](https://github.com/nataliakoliou/ML-Ciphertext-Decryption/blob/main/requirements.txt) and run ```pip install -r requirements.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

> The text file used in this project consists of short news articles in English from the Wall Street Journal: [wsj_untokenized.txt](https://github.com/nataliakoliou/ML-Ciphertext-Decryption/blob/main/requirements.txt). Please note that in order to run the code, you should have this text file in your local folder.

## Author
Natalia Koliou: find me on [LinkedIn](https://www.linkedin.com/in/natalia-k-b37b01197/).
