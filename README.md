# NLP-Various-Implementations
This repository contains all NLP course assignments.

## ```Assignment.1a``` | Tokenization using NLTK, SpaCy, and BERT
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


## Acknowledgments
I would like to express my gratitude to [Interactive Maths](https://crypto.interactive-maths.com/mixed-alphabet-cipher.html) for providing valuable information and resources, that contributed to the development of my project.

All of the books used as training & testing datasets in this project, were obtained from [Project Gutenberg](https://www.gutenberg.org/). Therefore, I would like to acknowledge its invaluable contribution in making these texts freely available for research and analysis.

## Conclusion
This code provides a basic implementation of an ML Ciphertext Decryption Algorithm. Users are encouraged to modify the training/testing datasets or the feature-tuple, to observe the impact on the total performance and accuracy.

## Author
Natalia Koliou: find me on [LinkedIn](https://www.linkedin.com/in/natalia-k-b37b01197/).
