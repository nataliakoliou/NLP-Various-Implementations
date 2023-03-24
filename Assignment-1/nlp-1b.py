import nltk
import math
from itertools import product
from nltk.corpus import treebank
from nltk.util import ngrams
from collections import Counter
from prettytable import PrettyTable

def download_treebank():
    if not nltk.data.find('corpora/treebank/combined'):
        nltk.download('treebank')

def split_corpus():
    corpus = treebank.fileids()
    train_corpus = corpus[:170] # 170 news files in train_corpus
    test_corpus = corpus[170:] # 29 news files in test_corpus
    return train_corpus, test_corpus           

def preprocess(corpus, vocab, start, end, label, lowercase):
    bigrams = []
    sents = [[start] + [w.lower() for w in sent] + [end] if lowercase else [start] + sent + [end] for sent in treebank.sents(corpus)]
    for sent in sents:
        padded_sent = ngrams(tuple(sent), 2, pad_left=True, pad_right=True, left_pad_symbol=start, right_pad_symbol=end) # pads the sentence with start and end symbols, and creates bigrams
        sent_bigrams = [(w1,w2) for (w1,w2) in padded_sent if (w1,w2) != (start, start) and (w1,w2) != (end,end)] # excludes bigrams where both words are the start or end symbols, and stores the remaining bigrams in a list
        bigrams.extend(sent_bigrams)
    bigrams = replace_tokens(bigrams, vocab, 3, label)
    return bigrams

def replace_tokens(bigrams, vocab, min_freq, label):
    word_counts = Counter([word for bigram in bigrams for word in bigram])
    if not vocab: return [tuple(label if word_counts[word] < min_freq else word for word in bigram) for bigram in bigrams]
    else: return [tuple(label if word not in vocab else word for word in bigram) for bigram in bigrams]

def create_vocab(train_bigrams):
    return {word for bigram in train_bigrams for word in bigram if word != '<UNK>'}

def train(k, vocab, bigrams):
    bigram_model = {}
    bigram_counts = Counter(bigrams) # counts the occurrences of each bigram (descending order counter object)
    words = sorted(set([word for bigram in bigrams for word in bigram])) # all unique words in the train_corpus
    for curr_word in words:
        total_count = sum([count for (first, second), count in bigram_counts.items() if first == curr_word])
        for next_word in words:
            count = bigram_counts[(curr_word, next_word)]
            bigram_model[(curr_word, next_word)] = (count + k) / (total_count + k*len(vocab))
    #biprint(bigram_model, words[:5])
    return bigram_model

def train(k, vocab, ngrams, n):
    ngram_model = {}
    ngram_counts = Counter(ngrams)
    words = sorted(set([word for ngram in ngrams for word in ngram]))
    for curr_words in product(words, repeat=n-1):
        total_count = sum([count for ngram, count in ngram_counts.items() if ngram[:-1] == curr_words])
        for next_word in words:
            count = ngram_counts[curr_words + (next_word,)]
            ngram_model[curr_words + (next_word,)] = (count + k) / (total_count + k*len(vocab))
    return ngram_model

def evaluate(bigram_model, bigrams):
    total_log_prob = sum(math.log(bigram_model[bigram]) for bigram in bigrams)
    # total prob για 1 bigram (row) = 1??
    perplexity = math.exp(-(total_log_prob/len(bigrams)))
    return perplexity

def biprint(bigram_table, words):
    pt = PrettyTable([''] + [f'\033[1m\033[4m{w}\033[0m' for w in words])
    pt.add_rows([[f'\033[1m\033[4m{curr_word}\033[0m'] + [f'{bigram_table[(curr_word, next_word)]:.6f}' for next_word in words] for curr_word in words])
    print(pt)
            
def main():
    a, b = 1, 0.01
    download_treebank()
    train_corpus, test_corpus = split_corpus()

    train_bigrams_0, train_bigrams_1 = preprocess(train_corpus, [], "<BOS>", "<EOS>", "<UNK>", False), preprocess(train_corpus, [], "<BOS>", "<EOS>", "<UNK>", True)
    vocab_0, vocab_1 = create_vocab(train_bigrams_0), create_vocab(train_bigrams_1)
    test_bigrams_0, test_bigrams_1 = preprocess(test_corpus, vocab_0, "<BOS>", "<EOS>", "<UNK>", False), preprocess(test_corpus, vocab_1, "<BOS>", "<EOS>", "<UNK>", True)

    # Bigram Models with k = 1 Smoothing, where 0: lowercase = False and 1: lowercase = True
    bigram_model_a0, bigram_model_a1 = train(a, vocab_0, train_bigrams_0), train(a, vocab_1, train_bigrams_1)
    bi_perp_a0, bi_perp_a1 = evaluate(bigram_model_a0, test_bigrams_0), evaluate(bigram_model_a1, test_bigrams_1)
    print("normal: ", bi_perp_a0)
    print("lowercase: ", bi_perp_a1)

    # Bigram Models with k = 0.01 Smoothing, where 0: lowercase = False and 1: lowercase = True
    bigram_model_b0, bigram_model_b1 = train(b, vocab_0, train_bigrams_0), train(b, vocab_1, train_bigrams_1)
    bi_perp_b0, bi_perp_b1 = evaluate(bigram_model_b0, test_bigrams_0), evaluate(bigram_model_b1, test_bigrams_1)
    print("normal: ", bi_perp_b0)
    print("lowercase: ", bi_perp_b1)
    
if __name__ == "__main__":
    main()
