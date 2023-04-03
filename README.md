# NLP-Various-Implementations
> **This repository contains all NLP course assignments.**

## Assignment.1a | Tokenization using NLTK, SpaCy, and BERT
This project provides a Python implementation of tokenization and frequency analysis on text data using three different libraries: NLTK, SpaCy, and Transformers. It includes functions to tokenize text using each of these libraries and then analyze the frequency of the resulting tokens, providing tables and graphs to visualize the results.

**In this part of the project we:**
* implemented three tokenization methods using NLTK's word_tokenize(), spaCy's en_core_web_sm model, and HuggingFace's BertTokenizer (bert-base-cased version).
* analyzed a file containing short news texts from the Wall Street Journal.
* then applied tokenization using these three different methods: nltk.word_tokenize(), en_core_web_sm English language model from spaCy, and BertTokenizer from HuggingFace.
* reported for each method, the total number of tokens, the number of different tokens found, a random sentence with its token list, a table of the 20 most frequent tokens with their frequency, probability of appearance, and product of position times probability.
* analyzed the percentage of tokens that appear once, twice, and thrice and compare it to Zipf's Law.
* find the best constant A that fits Zipf's Law for this set of texts and created a chart showing the predictions of Zipf's Law and the actual measurements.

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

**Alternatively:** you can download [requirements-1a.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-1/Assignment-1a/requirements-1a.txt) and run ```pip install -r requirements-1a.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** The text file used in this project consists of short news articles in English from the Wall Street Journal: [wsj_untokenized.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-1/Assignment-1a/wsj_untokenized.txt). Please note that in order to run the code, you should have this text file in your local folder.

---
## Assignment.1b | N-gram Language Models
This project provides a Python implementation of an n-gram language model that can generate text based on the patterns found in a given corpus. It uses a probabilistic approach to predict the likelihood of the next word in a sequence, based on the previous n-1 words. The model can be trained on any text data, allowing it to capture the nuances of different writing styles and genres. It also includes functions to generate text based on the trained model, allowing for the creation of new and unique writing.

**In this part of the project we:**
* implemented a natural language processing algorithm using N-grams.
* generated sentences by predicting the next word, based on the history of previous words, using a probability distribution learned from a corpus of training text.
* trained eight distinct N-gram models (bigram and trigram models for k={1,0.01}) and evaluated them, by measuring their perplexity.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* NumPy: ```pip install numpy```
* NLTK: ```pip install nltk```
* Matplotlib: ```pip install matplotlib```
* PrettyTable: ```pip install prettytable```

**Alternatively:** you can download [requirements-1b.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-1/Assignment-1b/requirements-1b.txt) and run ```pip install -r requirements-1b.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** The text file used in this project consists of short news articles in English from the Wall Street Journal: [wsj_untokenized.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-1/Assignment-1b/wsj_untokenized.txt). Please note that in order to run the code, you should have this text file in your local folder.

## Author
Natalia Koliou: find me on [LinkedIn](https://www.linkedin.com/in/natalia-k-b37b01197/).
