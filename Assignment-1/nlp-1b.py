import nltk
import math
from itertools import product
from nltk.corpus import treebank
from nltk.util import ngrams
from collections import Counter
from prettytable import PrettyTable

def download_treebank():
    if not nltk.data.find("corpora/treebank/combined"):
        nltk.download("treebank")

def split_corpus():
    corpus = treebank.fileids()
    train_corpus = corpus[:170] # 170 news files in train_corpus
    test_corpus = corpus[170:] # 29 news files in test_corpus
    return train_corpus, test_corpus        

def edit_corpus(corpus):
    corpus_0 = treebank.sents(corpus)
    corpus_1 = [[word.lower() for word in sent] for sent in corpus_0]
    return corpus_0, corpus_1

def preprocess(corpus, vocab, start, end, label, n):
    ngrams_list = []
    sents = [[start] + sent + [end] for sent in corpus]
    for sent in sents:
        padded_sent = ngrams(tuple(sent), n, pad_left=True, pad_right=True, left_pad_symbol=start, right_pad_symbol=end) # pads the sentence with start and end symbols, and creates ngrams of order n
        sent_ngrams = [ngram for ngram in padded_sent if ngram.count(start) < 2 and ngram.count(end) < 2]
        ngrams_list.extend(sent_ngrams)
    ngrams_list = replace_tokens(ngrams_list, vocab, start, end, label)
    return ngrams_list

def replace_tokens(ngrams, vocab, start, end, label):
    return [tuple(label if word not in vocab and word not in {start, end} else word for word in ngram) for ngram in ngrams]

def create_vocab(train_corpus, min_freq):
    return {word for word, count in Counter(word for sent in train_corpus for word in sent).items() if count >= min_freq}

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

def evaluate(ngram_model, ngrams):
    total_log_prob = sum(math.log(ngram_model[ngram]) for ngram in ngrams)
    perplexity = math.exp(-(total_log_prob/len(ngrams)))
    return perplexity

def build_and_apply(n, k, train_ngrams, vocab, test_ngrams):
    ngram_model = train(k, vocab, train_ngrams, n)
    perplexity = evaluate(ngram_model, test_ngrams)
    print(perplexity)

def biprint(bigram_table, words):
    pt = PrettyTable([""] + [f"\033[1m\033[4m{w}\033[0m" for w in words])
    pt.add_rows([[f"\033[1m\033[4m{curr_word}\033[0m"] + [f"{bigram_table[(curr_word, next_word)]:.6f}" for next_word in words] for curr_word in words])
    print(pt)

def pretty_print(bpa0, bpa1, bpb0, bpb1, tpa0, tpa1, tpb0, tpb1):
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Model", "k", "Lowercase", "Perplexity"]])
    pt.add_rows([["Bigram", 1, "False", bpa0], ["Bigram", 1, "True", bpa1], ["Bigram", 0.01, "False", bpb0], ["Bigram", 0.01, "True", bpb1],
                 ["Trigram", 1, "False", tpa0], ["Trigram", 1, "True", tpa1], ["Trigram", 0.01, "False", tpb0], ["Trigram", 0.01, "True", tpb1]])
    print(pt)

######################################################################################################################################################################

a, b, bi, tri, min_freq = 1, 0.01, 2, 3, 3
download_treebank()
train_corpus, test_corpus = split_corpus()
    
train_corpus_0, train_corpus_1 = edit_corpus(train_corpus); test_corpus_0, test_corpus_1 = edit_corpus(test_corpus)
vocab_0, vocab_1 = create_vocab(train_corpus_0, min_freq), create_vocab(train_corpus_1, min_freq)

"""
train_bigrams_0 = preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi)
train_bigrams_1 = preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi)
train_trigrams_0 = preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri)
train_trigrams_1 = preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri)
test_bigrams_0 = preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi)
test_bigrams_1 = preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi)
test_trigrams_0 = preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri)
test_trigrams_1 = preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri)
""" 

# Bigram Model with k = 1 Smoothing, where 0: lowercase = False
build_and_apply(bi, a, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi))

# Bigram Model with k = 1 Smoothing, where 1: lowercase = True
build_and_apply(bi, a, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi))

# Bigram Model with k = 0.01 Smoothing, where 0: lowercase = False
build_and_apply(bi, b, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi))

# Bigram Model with k = 0.01 Smoothing, where 1: lowercase = True
build_and_apply(bi, b, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi))

# Trigram Model with k = 1 Smoothing, where 0: lowercase = False
build_and_apply(tri, a, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri))

# Trigram Model with k = 1 Smoothing, where 1: lowercase = True
build_and_apply(tri, a, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri))

# Trigram Model with k = 0.01 Smoothing, where 0: lowercase = False
build_and_apply(tri, b, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri))

# Trigram Model with k = 0.01 Smoothing, where 1: lowercase = True
build_and_apply(tri, b, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri))

#pretty_print(bi_perp_a0, bi_perp_a1,  bi_perp_b0, bi_perp_b1, tri_perp_a0, tri_perp_a1, tri_perp_b0, tri_perp_b1)
