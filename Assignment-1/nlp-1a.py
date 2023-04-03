import nltk
import spacy
import transformers
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from prettytable import PrettyTable

def graph(x, *ys, title, xlabel, ylabel, labels):
    for i, y in enumerate(ys):
        plt.plot(x, y, label=labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def read(file):
    with open(file, 'r') as f:
        text = f.read()
    return text

def get_freq_table(tokens):
    return Counter(tokens)

def get_freq_values(frequencies):
    return [len([token for token in frequencies.keys() if frequencies[token] == f]) for f in [1, 2, 3]]

def zipf_estimate(f):
    return 1/(f*(f+1))

def tokenize(text, display):
    
    nltk_tokens = nltk_tokenize(text)
    spacy_tokens = spacy_tokenize(text)
    bert_tokens = bert_tokenize(text)
    
    table = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["NLTK", "SpaCy", "BERT"]])
    for i in range(max(map(len, [nltk_tokens, spacy_tokens, bert_tokens]))):
        table.add_row([t[i] if i < len(t) else '' for t in [nltk_tokens, spacy_tokens, bert_tokens]])
    print(table) if display else None

    return nltk_tokens, spacy_tokens, bert_tokens
    
def nltk_tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def spacy_tokenize(text):
    tz = spacy.load('en_core_web_sm') # loads English tokenizer, tagger, parser and NER
    doc = tz(text)
    tokens = [token.text for token in doc]
    return tokens

def bert_tokenize(text):
    tz = transformers.BertTokenizer.from_pretrained('bert-base-cased') # loads the BertTokenizer with the bert-base-cased model
    tokens = tz.tokenize(text)
    return tokens

def count(nltk_tokens, spacy_tokens, bert_tokens):
    table = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Method", "Tokens", "Types"]])
    table.add_rows([["NLTK", len(nltk_tokens), len(set(nltk_tokens))],
                    ["SpaCy", len(spacy_tokens), len(set(spacy_tokens))],
                    ["BERT", len(bert_tokens), len(set(bert_tokens))]])
    table.align = "r"
    print(table)
    
def get_statistics(nltk_tokens, spacy_tokens, bert_tokens):
    
    nltk_freq = get_freq_table(nltk_tokens)
    spacy_freq = get_freq_table(spacy_tokens)
    bert_freq = get_freq_table(bert_tokens)
    
    nltk_total = len(nltk_tokens)
    spacy_total = len(spacy_tokens)
    bert_total = len(bert_tokens)
    
    table_nltk = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Token", "Frequency", "Probability", "Zipf Product"]])
    for j, (token, frequency) in enumerate(nltk_freq.most_common(20), start=1):
        prob = frequency / nltk_total
        prod = j * prob
        table_nltk.add_row([token, frequency, round(prob,5), round(prod,5)])
        
    table_spacy = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Token", "Frequency", "Probability", "Zipf Product"]])
    for j, (token, frequency) in enumerate(spacy_freq.most_common(20), start=1):
        prob = frequency / spacy_total
        prod = j * prob
        table_spacy.add_row([token, frequency, round(prob,5), round(prod,5)])
        
    table_bert = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Token", "Frequency", "Probability", "Zipf Product"]])
    for j, (token, frequency) in enumerate(bert_freq.most_common(20), start=1):
        prob = frequency / bert_total
        prod = j * prob
        table_bert.add_row([token, frequency, round(prob,5), round(prod,5)])

    for name, table in zip(["NLTK", "SpaCy", "BERT"], [table_nltk, table_spacy, table_bert]):
        print('\033[1m' + f"\n{name} Statistics:" + '\033[0m')
        print(table)
    
def get_occurancies(nltk_tokens, spacy_tokens, bert_tokens):
    
    nltk_freq = get_freq_table(nltk_tokens)
    spacy_freq = get_freq_table(spacy_tokens)
    bert_freq = get_freq_table(bert_tokens)

    nltk_once, nltk_twice, nltk_thrice = get_freq_values(nltk_freq)
    spacy_once, spacy_twice, spacy_thrice = get_freq_values(spacy_freq)
    bert_once, bert_twice, bert_thrice = get_freq_values(bert_freq)

    nltk_total = len(nltk_tokens)
    spacy_total = len(spacy_tokens)
    bert_total = len(bert_tokens)
    
    nltk_y = [round(nltk_once/nltk_total*100, 2), round(nltk_twice/nltk_total*100, 2), round(nltk_thrice/nltk_total*100, 2)]
    spacy_y = [round(spacy_once/spacy_total*100, 2), round(spacy_twice/spacy_total*100, 2), round(spacy_thrice/spacy_total*100, 2)]
    bert_y = [round(bert_once/bert_total*100, 2), round(bert_twice/bert_total*100, 2), round(bert_thrice/bert_total*100, 2)]
    zipf_y = [round(zipf_estimate(1)*100, 2), round(zipf_estimate(2)*100, 2), round(zipf_estimate(3)*100, 2)]
    
    table = PrettyTable(field_names=[f"\033[1m{'Method'}\033[0m", f"\033[1m{'Once'}\033[0m", f"\033[1m{'Twice'}\033[0m", f"\033[1m{'Thrice'}\033[0m"])
    table.add_rows([["NLTK", f"{nltk_y[0]}%", f"{nltk_y[1]}%", f"{nltk_y[2]}%"],
                    ["SpaCy", f"{spacy_y[0]}%", f"{spacy_y[1]}%", f"{spacy_y[2]}%"],
                    ["BERT", f"{bert_y[0]}%", f"{bert_y[1]}%", f"{bert_y[2]}%"],
                    ["Zipf", f"{zipf_y[0]}%", f"{zipf_y[1]}%", f"{zipf_y[2]}%"]])
    table.align = "l"
    print(table)
    graph([1, 2, 3], nltk_y, spacy_y, bert_y, zipf_y, title="Token Frequency Estimations", xlabel="Token Frequency", ylabel="Percentage", labels=["NLTK", "SpaCy", "BERT", "Zipf"])

def find_optimal_A(nltk_tokens, spacy_tokens, bert_tokens):

    nltk_prob = {token: freq/len(nltk_tokens) for token, freq in get_freq_table(nltk_tokens).items()}
    spacy_prob = {token: freq/len(spacy_tokens) for token, freq in get_freq_table(spacy_tokens).items()}
    bert_prob = {token: freq/len(bert_tokens) for token, freq in get_freq_table(bert_tokens).items()}

    A = {}
    for method, prob_table in [('nltk', nltk_prob), ('spacy', spacy_prob), ('bert', bert_prob)]:
        product_sum = 0
        for rank, (token, prob) in enumerate(Counter(prob_table).most_common(), start=1):
            product_sum += prob * rank
        A[method] = product_sum / len(prob_table)
    table = PrettyTable(field_names=[f"\033[1m{'Method'}\033[0m", f"\033[1m{'Optimal A'}\033[0m"])
    table.add_rows([["NLTK", round(A['nltk'],7)], ["SpaCy", round(A['spacy'],7)], ["BERT", round(A['bert'],7)]])
    table.align = "l"
    print(table)
    return A

def draw_distribution(A, nltk_tokens, spacy_tokens, bert_tokens):
    
    nltk_freq = get_freq_table(nltk_tokens).most_common()
    spacy_freq = get_freq_table(spacy_tokens).most_common()
    bert_freq = get_freq_table(bert_tokens).most_common()
    
    nltk_x, nltk_y = zip(*enumerate([freq for token, freq in nltk_freq], start=1))
    spacy_x, spacy_y = zip(*enumerate([freq for token, freq in spacy_freq], start=1))
    bert_x, bert_y = zip(*enumerate([freq for token, freq in bert_freq], start=1))
    
    nltk_zipf = [A['nltk']*len(nltk_tokens)/rate for rate in nltk_x]
    spacy_zipf = [A['spacy']*len(spacy_tokens)/rate for rate in spacy_x]
    bert_zipf = [A['bert']*len(bert_tokens)/rate for rate in bert_x]
    
    graph(np.log10(nltk_x), np.log10(nltk_y), np.log10(nltk_zipf), title="NLTK Token Frequency Distribution", xlabel="Token Rank", ylabel="Token Frequency", labels=["NLTK", "Zipf"])
    graph(np.log10(spacy_x), np.log10(spacy_y), np.log10(spacy_zipf), title="SpaCy Token Frequency Distribution", xlabel="Token Rank", ylabel="Token Frequency", labels=["SpaCy", "Zipf"])
    graph(np.log10(bert_x), np.log10(bert_y), np.log10(bert_zipf), title="BERT Token Frequency Distribution", xlabel="Token Rank", ylabel="Token Frequency", labels=["BERT", "Zipf"])

##########################################################################################################################################################################################
    
text = read('wsj_untokenized.txt')

t1a, t2a, t3a = tokenize(text, False)
count(t1a, t2a, t3a)

sentence = " ''Cosby'' is down a full ratings point in the week of Oct. 2-8 over the same week a year ago, according to A.C. Nielsen Co. Mr. Gillespie at Viacom says the ratings are rising."
t1b, t2b, t3b = tokenize(sentence, True)

get_statistics(t1a, t2a, t3a)

get_occurancies(t1a, t2a, t3a)

alphas = find_optimal_A(t1a, t2a, t3a)
draw_distribution(alphas, t1a, t2a, t3a)
