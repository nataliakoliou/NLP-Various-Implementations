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

---
## Assignment.2a | Word Embeddings Similarities & Analogies
This project provides a Python implementation of word embeddings using the pre-trained word2vec and GloVe models. Word embeddings are a way of representing words in a multidimensional vector space, where similar words are closer together, and different words are further apart. Using word embeddings, we can perform various tasks such as measuring the similarity between words or finding analogies. In this project, we use the word2vec and GloVe models to find similar words and analogies. Specifically, we implement a natural language processing algorithm that uses the pre-trained models to find the most similar words to a given set of words or to solve analogies such as "man is to king as woman is to what?".

**In this part of the project we:**
* found the most similar words to a given set of words using the pre-trained word embeddings word2vec and GloVe.
* defined analogies by solving the equation "A is to B as C is to D" using these word2vec and GloVe models.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* Gensim: ```pip install gensim```
* PrettyTable: ```pip install prettytable```

**Alternatively:** you can download [requirements-2a.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-2/Assignment-2a/requirements-2a.txt) and run ```pip install -r requirements-2a.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

---
## Assignment.2b | Text Classification using TF-IDF Vectorization
This project provides a Python implementation of a text classification algorithm using TF-IDF vectorization. This algorithm extracts features from text data using n-grams and trains different classifiers to predict the labels of new texts. It makes use of the TfidfVectorizer and machine learning libraries like MultinomialNB and LinearSVC from scikit-learn to achieve this.

**In this part of the project we:**
* preprocessed the data by concatenating columns and separating labels from features, and extracted n-gram features from them using the TfidfVectorizer module.
* trained several machine learning models (either MultinomialNB or LinearSVC) using the features and labels.
* evaluated the models' performance on a test set by predicting the labels and calculating the accuracy score.
* visualized the models' performance and analyzed misclassified data to identify common misclassification patterns and vulnerabilities.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* Scikit-learn: ```pip install -U scikit-learn```
* PrettyTable: ```pip install prettytable```

**Alternatively:** you can download [requirements-2b.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-2/Assignment-2b/requirements-2b.txt) and run ```pip install -r requirements-2b.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** For this project, we used the AG News Classification Dataset which contains short news articles from four different categories. You can download the dataset from Kaggle using this link: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset.  Please note that in order to run the code, you should have these csv files: `train.csv` and `test.csv` in your local folder.

---
## Assignment.2c | Text Classification with Neural Networks
This project provides a Python implementation of text classification algorithms using neural networks, including RNNs and LSTMs. The project involves training several models with different architectures and hyperparameters on the AG News Topic Classification and the IMDB movie review datasets to evaluate their performance on simple classification tasks. Through experimenting with different architectures and hyperparameters, the project provides valuable hands-on experience in training neural networks and insights into the factors that affect their performance in text classification tasks. The project aims to provide a deeper understanding of how these models operate and can be optimized to achieve high accuracy levels.

**In this part of the project we:**
* trained and evaluated several neural network models for text classification using the AG News Topic Classification and IMDB movie review datasets.
* used pre-trained word embeddings and trained our own word embeddings using the Word2Vec algorithm.
* performed hyperparameter tuning and grid search to optimize the performance of our models.
* visualized the training and validation accuracy and loss to analyze the models' behavior during training.
* analyzed the models' performance and identified the most difficult samples to classify.
* compared the performance of different models and analyzed their strengths and weaknesses.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* Pandas: ```pip install pandas```
* PyTorch: ```pip install torch torchvision```
* Tqdm: ```pip install tqdm```
* Scikit-learn: ```pip install -U scikit-learn```
* Torchtext: ```pip install torchtext```
* PrettyTable: ```pip install prettytable```

**Alternatively:** you can download [requirements-2c.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-2/Assignment-2c/requirements-2c.txt) and run ```pip install -r requirements-2c.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** For this project, we used the AG News Classification Dataset which contains short news articles from four different categories and the IMDB movie review dataset which contains movie reviews from IMDB users, labeled either positive or negative. You can download the AG News dataset from Kaggle using this link: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset and the IMDB dataset from this link: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews. We also used pre-trained GloVe embeddings, which we downloaded from the Stanford NLP group website (https://nlp.stanford.edu/projects/glove/). Please note that in order to run the code, you should have these files: `train.csv`, `test.csv`, `IMDB Dataset.csv` and `glove.6B.100d.txt` in your local folder.

---
## Assignment.3 | Sequence Labeling with Pre-trained Language Models
This project provides a Python implementation of a tag-recognition algorithm, that uses pre-trained language models to assign NER, Chunk and POS tags to sentence tokens. This algorithm provides functionalities for loading sentences, creating tagsets, aligning labels, encoding data, counting model parameters, initializing models, setting up data loaders, evaluating model performance, training the model, and displaying results. It supports both BERT and RoBERTa models for token classification and allows customization of parameters such as epochs, batch size, learning rate, and device selection (CPU or GPU). The implementation includes training, validation, and testing capabilities, along with features for detecting misclassifications and generating classification reports.

**In this part of the project we:**
* developed a tag-recognition algorithm that utilizes pre-trained language models to assign NER, Chunk, and POS tags to sentence tokens.
* supported both BERT and RoBERTa models for token classification.
* enabled training, validation, and testing capabilities.
* detected tag misclassifications.
* evaluated ChatGPT's performance in NER, Chunk and POS tagging using zero-shot and few-shot prompting.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* PyTorch: ```pip install torch```
* Pandas: ```pip install pandas```
* Torchtext: ```pip install torchtext```
* Transformers: ```pip install transformers```
* Scikit-learn: ```pip install scikit-learn```
* Tqdm: ```pip install tqdm```

**Alternatively:** you can download [requirements-3.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-3/requirements-3.txt) and run ```pip install -r requirements-3.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** For this project, we used the For this project, we used the Conll003-Englishversion dataset for named-entity recognition (NER) using BERT. You can download the dataset from Kaggle using this link: https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion. We also used a single-sentence dataset we created ourselves: [example.txt](https://github.com/nataliakoliou/NLP-Various-Implementations/blob/main/Assignment-3/example.txt). Please note that in order to run the code, you should have these files: `train.txt`, `valid.txt`, `test.txt` and `example.txt` in your local folder.

---
## Assignment.4 | Transition-based vs Graph-based Dependency Parser
This project provides an evaluation of two dependency parser: a transition-based parser inspired by Chen & Manning and a graph-based parser inspired by Kiperwasser & Goldberg. Various modifications were made to both parsers, including changes in word embeddings, activation functions, and the introduction of pretrained language models like BERT. The performance of these parsers was measured using the Unlabeled Attachment Score (UAS) and sometimes the Labeled Attachment Score (LAS) on the test set, providing insights into the impact of different techniques and architectural choices on dependency parsing accuracy.

**In this part of the project we:**
* experimented with randomly initialized word embeddings instead of pretrained embeddings.
* explored the use of only word features and pretrained word embeddings.
* added an extra hidden layer with ReLU activation to the model.
* defined and applied the cubic activation function.
* replaced learned word embeddings with pretrained Glove-6B-100d embeddings.
* changed the activation function in the MLPs' hidden layer from tanh to ReLU.
* replaced the BiLSTM encoder with a pretrained BERT language model.

**```>```** Please note that running this particular section of the project locally is not feasible for users. The code presented here serves as an evaluation of pre-existing implementations, which have been selectively modified to investigate specific aspects. The intention is to offer observations and insights, rather than providing executable code.

## Author
Natalia Koliou: find me on [LinkedIn](https://www.linkedin.com/in/natalia-koliou-b37b01197/).
