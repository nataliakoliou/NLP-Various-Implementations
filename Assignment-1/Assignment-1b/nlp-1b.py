import nltk
import math
import random
from nltk.corpus import treebank
from nltk.util import ngrams
from collections import defaultdict, Counter
from prettytable import PrettyTable

def download_treebank():
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

def train(k, vocab, ngrams):
    ngram_model, ngram_counts = defaultdict(Counter), Counter(ngrams)
    prefix_counts, prefix_probs = defaultdict(int), defaultdict(float)
    for ngram, count in ngram_counts.items():
        prefix_counts[ngram[:-1]] += count
    for prefix, count in prefix_counts.items():
        prefix_probs[prefix] = k / (count + k * len(vocab))
    for ngram, count in ngram_counts.items():
        ngram_model[ngram[:-1]][ngram[-1]] = (count + k) / (prefix_counts[ngram[:-1]] + k * len(vocab))
    return ngram_model, prefix_probs

def evaluate(ngram_model, prefix_probs, ngrams, vocab):
    total_log_prob = 0.0
    total_log_prob = sum([math.log(ngram_model[ngram[:-1]].get(ngram[-1], prefix_probs[ngram[:-1]])) if ngram[:-1] in ngram_model else math.log(1/len(vocab)) for ngram in ngrams])
    perplexity = math.exp(-(total_log_prob/len(ngrams)))
    return perplexity

def build_and_apply(n, k, train_ngrams, vocab, test_ngrams, lowercase):
    ngram_model, prefix_probs = train(k, vocab, train_ngrams)
    perplexity = evaluate(ngram_model, prefix_probs, test_ngrams, vocab)
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Model", "k", "Lowercase", "Perplexity"]])
    pt.add_row(["Bigram", k, lowercase, perplexity]) if n == 2 else pt.add_row(["Trigram", k, lowercase, perplexity])
    print(pt)
    return ngram_model

def generate_sentences(model, n, start, end, vocab, num_sents):
    for ns in range(num_sents):
        start_ngrams = [ngram for ngram in model.keys() if ngram[0] == start]
        random_ngram = random.choice(start_ngrams)
        sentence = list(random_ngram[:n-1])
        while sentence[-1] != end:
            prefix = tuple(sentence[-n+1:])  # get the last n-1 words in sentence as a tuple
            if prefix not in model:
                sentence.append(end) if sentence[-1] != end else None
                break
            candidates = [(c, p) for c, p in model[prefix].items() if c in vocab or c == end]
            if not candidates:
                sentence.append(end) if sentence[-1] != end else None
                break
            choices, probabilities = zip(*candidates)
            sentence.append(random.choices(choices, weights=probabilities)[0])
        print(f"\033[1mSentence {ns+1}:\033[0m " + " ".join(sentence))

######################################################################################################################################################################

a, b, bi, tri, min_freq, ns = 1, 0.01, 2, 3, 3, 3
download_treebank()
train_corpus, test_corpus = split_corpus()
    
train_corpus_0, train_corpus_1 = edit_corpus(train_corpus); test_corpus_0, test_corpus_1 = edit_corpus(test_corpus)
vocab_0, vocab_1 = create_vocab(train_corpus_0, min_freq), create_vocab(train_corpus_1, min_freq)

# Bigram Model with k = 1 Smoothing, where 0: lowercase = False
bigram_model_a0 = build_and_apply(bi, a, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi), False)
generate_sentences(bigram_model_a0, bi, "<BOS>", "<EOS>", vocab_0, ns)

# Bigram Model with k = 1 Smoothing, where 1: lowercase = True
bigram_model_a1 = build_and_apply(bi, a, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi), True)
generate_sentences(bigram_model_a1, bi, "<BOS>", "<EOS>", vocab_1, ns)

# Bigram Model with k = 0.01 Smoothing, where 0: lowercase = False
bigram_model_b0 = build_and_apply(bi, b, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", bi), False)
generate_sentences(bigram_model_b0, bi, "<BOS>", "<EOS>", vocab_0, ns)

# Bigram Model with k = 0.01 Smoothing, where 1: lowercase = True
bigram_model_b1 = build_and_apply(bi, b, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", bi), True)
generate_sentences(bigram_model_b1, bi, "<BOS>", "<EOS>", vocab_1, ns)

# Trigram Model with k = 1 Smoothing, where 0: lowercase = False
trigram_model_a0 = build_and_apply(tri, a, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri), False)
generate_sentences(trigram_model_a0, tri, "<BOS>", "<EOS>", vocab_0, ns)

# Trigram Model with k = 1 Smoothing, where 1: lowercase = True
trigram_model_a1 = build_and_apply(tri, a, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri), True)
generate_sentences(trigram_model_a1, tri, "<BOS>", "<EOS>", vocab_1, ns)

# Trigram Model with k = 0.01 Smoothing, where 0: lowercase = False
trigram_model_b0 = build_and_apply(tri, b, preprocess(train_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri), vocab_0, preprocess(test_corpus_0, vocab_0, "<BOS>", "<EOS>", "<UNK>", tri), False)
generate_sentences(trigram_model_b0, tri, "<BOS>", "<EOS>", vocab_0, ns)

# Trigram Model with k = 0.01 Smoothing, where 1: lowercase = True
trigram_model_b1 = build_and_apply(tri, b, preprocess(train_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri), vocab_1, preprocess(test_corpus_1, vocab_1, "<BOS>", "<EOS>", "<UNK>", tri), True)
generate_sentences(trigram_model_b1, tri, "<BOS>", "<EOS>", vocab_1, ns)
