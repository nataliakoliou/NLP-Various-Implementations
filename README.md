# NLP-Various-Implementations
This repository contains all NLP course assignments.

## Assignment.1a | Tokenization using NLTK, SpaCy, and BERT
This project provides a Python implementation of tokenization and frequency analysis on text data using three different libraries: NLTK, SpaCy, and Transformers. It includes functions to tokenize text using each of these libraries and then analyze the frequency of the resulting tokens, providing tables and graphs to visualize the results.

**In this part of the project:**
* three tokenization methods were implemented using NLTK's word_tokenize(), spaCy's en_core_web_sm model, and HuggingFace's BertTokenizer (bert-base-cased version).
* a file containing short news texts from the Wall Street Journal was analyzed, and tokenization was applied using these three different methods: nltk.word_tokenize(), en_core_web_sm English language model from spaCy, and BertTokenizer from HuggingFace.
* for each method, the total number of tokens, the number of different tokens found, a random sentence with its token list, a table of the 20 most frequent tokens with their frequency, probability of appearance, and product of position times probability were reported.
* the percentage of tokens that appear once, twice, and thrice were also analyzed and compared to Zipf's Law.
* the best constant A that fits Zipf's Law for this set of texts was found, and a chart was created showing the predictions of Zipf's Law and the actual measurements.

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

## Assignment.1b | Tokenization using NLTK, SpaCy, and BERT
This project provides a Python implementation of tokenization and frequency analysis on text data using three different libraries: NLTK, SpaCy, and Transformers. It includes functions to tokenize text using each of these libraries and then analyze the frequency of the resulting tokens, providing tables and graphs to visualize the results.

**In this part of the project:**
* three tokenization methods were implemented using NLTK's word_tokenize(), spaCy's en_core_web_sm model, and HuggingFace's BertTokenizer (bert-base-cased version).
* a file containing short news texts from the Wall Street Journal was analyzed, and tokenization was applied using these three different methods: nltk.word_tokenize(), en_core_web_sm English language model from spaCy, and BertTokenizer from HuggingFace.
* for each method, the total number of tokens, the number of different tokens found, a random sentence with its token list, a table of the 20 most frequent tokens with their frequency, probability of appearance, and product of position times probability were reported.
* the percentage of tokens that appear once, twice, and thrice were also analyzed and compared to Zipf's Law.
* the best constant A that fits Zipf's Law for this set of texts was found, and a chart was created showing the predictions of Zipf's Law and the actual measurements.

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
