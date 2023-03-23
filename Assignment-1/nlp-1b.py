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

def preprocess(train_corpus, test_corpus, start, end, lowercase):
    train_bigrams = []
    train_sents = [[start] + [w.lower() for w in sent] + [end] if lowercase else [start] + sent + [end] for sent in treebank.sents(train_corpus)]
    test_sents = [[start] + [w.lower() for w in sent] + [end] if lowercase else [start] + sent + [end] for sent in treebank.sents(test_corpus)]
    for sent in train_sents:
        padded_sent = ngrams(tuple(sent), 2, pad_left=True, pad_right=True, left_pad_symbol=start, right_pad_symbol=end) # pads the sentence with start and end symbols, and creates bigrams
        sent_bigrams = [(w1,w2) for (w1,w2) in padded_sent if (w1,w2) != (start, start) and (w1,w2) != (end,end)] # excludes bigrams where both words are the start or end symbols, and stores the remaining bigrams in a list
        train_bigrams.extend(sent_bigrams)
    train_bigrams = replace_tokens(train_bigrams, 3, "<UNK>")
    return train_sents, test_sents, train_bigrams

def replace_tokens(bigrams, min_freq, label):
    word_counts = Counter([word for bigram in bigrams for word in bigram])
    return [tuple(label if word_counts[word] < min_freq else word for word in bigram) for bigram in bigrams]

def create_vocab(train_bigrams):
    return {word for bigram in train_bigrams for word in bigram if word != '<UNK>'}

def get_bigrams_freqs(train_bigrams):
    bigram_counts = Counter(train_bigrams) # counts the occurrences of each bigram (descending order counter object)
    words = sorted(set([word for bigram in train_bigrams for word in bigram])) # all unique words in the train_corpus
    bigram_table = {word: {next_word: 0 for next_word in words} for word in words} # initializes the bigram table with zeros
    for bigram, count in bigram_counts.items(): # e.x: bigram_counts.items() = [(('a', 'cat'), 2), (('cat', 'in'), 1)]
        bigram_table[bigram[0]][bigram[1]] += count
    
    """
    pt = PrettyTable(field_names=[''] + ['\033[1m\033[4m' + w + '\033[0m' for w in words])
    pt.add_rows([['\033[1m\033[4m' + word + '\033[0m'] + [f'{bigram_table[word][next_word]:.2f}' for next_word in bigram_table[word]] for word in bigram_table])
    print(pt)
    """

def main():
    download_treebank()
    train_corpus, test_corpus = split_corpus()
    train_sents, test_sents, train_bigrams = preprocess(train_corpus, test_corpus, "<BOS>", "<EOS>", False) # 3509 sentences with 94050 bigrams in train_corpus
    vocab = create_vocab(train_bigrams) # 5565 unique known words in train_bigrams
    get_bigrams_freqs(train_bigrams)
    
if __name__ == "__main__":
    main()
