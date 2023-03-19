import nltk
from nltk.corpus import treebank
from nltk.lm import Vocabulary
from nltk.lm.models import Laplace
from nltk.util import ngrams

def main():

    if not nltk.data.find('corpora/treebank/combined'):
        nltk.download('treebank')  # downloads the treebank resource from NLTK

    corpus = treebank.fileids() # loads the Wall Street Journal corpus

    # split the corpus into train and test sets
    train_corpus = corpus[:170]
    test_corpus = corpus[170:]
    
    start = "<BOS>"
    end = "<EOS>"
    unknown = "<UNK>" 

    # get the sentences for each set and add start/end symbols
    train_sentences = [[start] + sent + [end] for sent in treebank.sents(train_corpus)]
    train_sentences_flat = [word for sentence in train_sentences for word in sentence]
    train_sentences_ngrams = [[(w1,w2) for (w1,w2) in sent if (w1,w2) != ('<BOS>', '<BOS>') and (w1,w2) != ('<EOS>', '<EOS>')] for sent in [ngrams(tuple(sentence), 2, pad_left=True, pad_right=True, left_pad_symbol=start, right_pad_symbol=end) for sentence in train_sentences]]
    
    #print([i for i in train_sentences_ngrams[0]])
    #print([i for j in range(len(train_sentences_ngrams)) for i in train_sentences_ngrams[j]])
    
    test_sentences = [[start] + sent + [end] for sent in treebank.sents(test_corpus)]

    # create the vocabulary for the model
    vocab = Vocabulary(train_sentences_flat, unk_cutoff=3, unk_label=unknown)
    # map each word in each sentence of test_sentences to its corresponding index in the vocabulary created from the training corpus and replace out-of-vocabulary words with the <UNK> token during evaluation
    test_sentences = [[vocab.lookup(token) if vocab.lookup(token) is not None else vocab.lookup(unknown) for token in sent] for sent in test_sentences]
    
    # create the Laplace (add-one smoothing) model and train it on the training set
    laplace_model = Laplace(order=2, vocabulary=vocab)
    laplace_model.fit(train_sentences_ngrams)
        
    # evaluate the model on the test set
    test_perplexity = laplace_model.perplexity(test_sentences)
    print("Test perplexity:", test_perplexity)
    
if __name__ == "__main__":
    main()
