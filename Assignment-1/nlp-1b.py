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

def preprocess(corpus, vocab, start, end, label, lowercase, n):
    ngrams_list = []
    sents = [[start] + [w.lower() for w in sent] + [end] if lowercase else [start] + sent + [end] for sent in treebank.sents(corpus)]
    for sent in sents:
        padded_sent = ngrams(tuple(sent), n, pad_left=True, pad_right=True, left_pad_symbol=start, right_pad_symbol=end) # pads the sentence with start and end symbols, and creates ngrams of order n
        sent_ngrams = [ngram for ngram in padded_sent if ngram.count(start) < 2 and ngram.count(end) < 2]
        ngrams_list.extend(sent_ngrams)
    ngrams_list = replace_tokens(ngrams_list, vocab, 3, label)
    return ngrams_list

def replace_tokens(ngrams, vocab, min_freq, label):
    word_counts = Counter([word for ngram in ngrams for word in ngram])
    if not vocab: return [tuple(label if word_counts[word] < min_freq else word for word in ngram) for ngram in ngrams]
    else: return [tuple(label if word not in vocab else word for word in ngram) for ngram in ngrams]

def create_vocab(train_ngrams, label):
    return {word for ngram in train_ngrams for word in ngram if word != label}

def train(k, vocab, ngrams):
    ngram_model = {}
    ngram_counts = Counter(ngrams)
    words = sorted(set([word for ngram in ngrams for word in ngram]))
    for ngram in ngrams:
        curr_words = ngram[:-1]  # extracts the first n-1 words from the n-gram and assigns them to curr_words (sliding window)
        total_count = sum([count for ngram, count in ngram_counts.items() if ngram[:-1] == curr_words])
        for next_word in get_candidate_words(words, ngram_counts, curr_words):
            count = ngram_counts[curr_words + (next_word,)]
            ngram_model[curr_words + (next_word,)] = (count + k) / (total_count + k*len(vocab))
    return ngram_model

def get_candidate_words(words, ngram_counts, curr_words):
    candidate_words = set()
    for next_word in words:
        if ngram_counts.get(curr_words + (next_word,), 0) > 0:
            candidate_words.add(next_word)
    return candidate_words

def evaluate(ngram_model, ngrams):
    total_log_prob = sum(math.log(ngram_model[ngram]) for ngram in ngrams)
    perplexity = math.exp(-(total_log_prob/len(ngrams)))
    return perplexity

def evaluate(k, vocab, ngram_model, ngrams):
    total_log_prob = 0
    for ngram in ngrams:
        if ngram in ngram_model:
            total_log_prob += math.log(ngram_model[ngram])
        else:
            total_log_prob += k / (total_count + k*len(vocab))
    perplexity = math.exp(-(total_log_prob/len(ngrams)))
    return perplexity

def biprint(bigram_table, words):
    pt = PrettyTable([""] + [f"\033[1m\033[4m{w}\033[0m" for w in words])
    pt.add_rows([[f"\033[1m\033[4m{curr_word}\033[0m"] + [f"{bigram_table[(curr_word, next_word)]:.6f}" for next_word in words] for curr_word in words])
    print(pt)

def pretty_print(bpa0, bpa1, bpb0, bpb1, tpa0, tpa1, tpb0, tpb1):
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Model", "k", "Lowercase", "Perplexity"]])
    pt.add_rows([["Bigram", 1, "False", bpa0], ["Bigram", 1, "True", bpa1], ["Bigram", 0.01, "False", bpb0], ["Bigram", 0.01, "True", bpb1],
                 ["Trigram", 1, "False", tpa0], ["Trigram", 1, "True", tpa1], ["Trigram", 0.01, "False", tpb0], ["Trigram", 0.01, "True", tpb1]])
    print(pt)

def main():
    a, b = 1, 0.01
    download_treebank()
    train_corpus, test_corpus = split_corpus()

    train_bigrams_0, train_bigrams_1, train_trigrams_0, train_trigrams_1 = preprocess(train_corpus, [], "<BOS>", "<EOS>", "<UNK>", False, 2), preprocess(train_corpus, [], "<BOS>", "<EOS>", "<UNK>", True, 2), preprocess(train_corpus, [], "<BOS>", "<EOS>", "<UNK>", False, 3), preprocess(train_corpus, [], "<BOS>", "<EOS>", "<UNK>", True, 3)
    vocab_0, vocab_1 = create_vocab(train_bigrams_0, "<UNK>"), create_vocab(train_bigrams_1, "<UNK>")
    test_bigrams_0, test_bigrams_1, test_trigrams_0, test_trigrams_1 = preprocess(test_corpus, vocab_0, "<BOS>", "<EOS>", "<UNK>", False, 2), preprocess(test_corpus, vocab_1, "<BOS>", "<EOS>", "<UNK>", True, 2), preprocess(test_corpus, vocab_0, "<BOS>", "<EOS>", "<UNK>", False, 3), preprocess(test_corpus, vocab_1, "<BOS>", "<EOS>", "<UNK>", True, 3)

    # Bigram Model with k = 1 Smoothing, where 0: lowercase = False and 1: lowercase = True
    bigram_model_a0, bigram_model_a1 = train(a, vocab_0, train_bigrams_0), train(a, vocab_1, train_bigrams_1)
    bi_perp_a0, bi_perp_a1 = evaluate(bigram_model_a0, test_bigrams_0), evaluate(bigram_model_a1, test_bigrams_1)
    print(bi_perp_a0)

    # Bigram Model with k = 0.01 Smoothing, where 0: lowercase = False and 1: lowercase = True
    #bigram_model_b0, bigram_model_b1 = train(b, vocab_0, train_bigrams_0), train(b, vocab_1, train_bigrams_1)
    #bi_perp_b0, bi_perp_b1 = evaluate(bigram_model_b0, test_bigrams_0), evaluate(bigram_model_b1, test_bigrams_1)

    # Trigram Model with k = 1 Smoothing, where 0: lowercase = False and 1: lowercase = True
    #trigram_model_a0 = train(a, vocab_0, train_trigrams_0)
    #tri_perp_a0 = evaluate(trigram_model_a0, test_trigrams_0)
    #print(tri_perp_a0)

    #trigram_model_a1 = train(a, vocab_1, train_trigrams_1)
    #tri_perp_a1 = evaluate(trigram_model_a1, test_trigrams_1)

    # Trigram Model with k = 0.01 Smoothing, where 0: lowercase = False and 1: lowercase = True
    #trigram_model_b0, trigram_model_b1 = train(b, vocab_0, train_trigrams_0), train(b, vocab_1, train_trigrams_1)
    #tri_perp_b0, tri_perp_b1 = evaluate(trigram_model_b0, test_trigrams_0), evaluate(trigram_model_b1, test_trigrams_1)

    #pretty_print(bi_perp_a0, bi_perp_a1,  bi_perp_b0, bi_perp_b1, tri_perp_a0, tri_perp_a1, tri_perp_b0, tri_perp_b1)

if __name__ == "__main__":
    main()
