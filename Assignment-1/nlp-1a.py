import nltk
import spacy
import transformers
import matplotlib.pyplot as plt
from collections import Counter
from prettytable import PrettyTable

def graph(x, *ys, labels):
    for i, y in enumerate(ys):
        plt.plot(x, y, label=labels[i])
    plt.title("Token Frequency Estimations")
    plt.xlabel("Token Frequency")
    plt.ylabel("Percentage")
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
    
    if display:
        print('\033[1m'+"\nNLTK:"+'\033[0m', nltk_tokens)
        print('\033[1m'+"\nSpaCy:"+'\033[0m', spacy_tokens)
        print('\033[1m'+"\nBERT:"+'\033[0m', bert_tokens)
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
    
    table = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Method", "Token", "Frequency", "Probability", "Zipf Product"]])
    for i, (method, freq) in enumerate([("NLTK", nltk_freq), ("SpaCy", spacy_freq), ("BERT", bert_freq)]):
        for j, (token, frequency) in enumerate(freq.most_common(20), start=1):
            prob = frequency / (nltk_total if method == "NLTK" else spacy_total if method == "SpaCy" else bert_total)
            prod = j * prob
            table.add_row([method, token, frequency, round(prob,5), round(prod,5)])
        if i != 2:
            table.add_row(["......", ".....", ".........", "...........", "............"])
            
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
    
    table = PrettyTable(field_names=[f"\033[1m{'Method'}\033[0m", f"\033[1m{'Singletons'}\033[0m", f"\033[1m{'Doubletons'}\033[0m", f"\033[1m{'Triplets'}\033[0m"])
    table.add_rows([["NLTK", f"{nltk_y[0]}%", f"{nltk_y[1]}%", f"{nltk_y[2]}%"],
                    ["SpaCy", f"{spacy_y[0]}%", f"{spacy_y[1]}%", f"{spacy_y[2]}%"],
                    ["BERT", f"{bert_y[0]}%", f"{bert_y[1]}%", f"{bert_y[2]}%"],
                    ["Zipf", f"{zipf_y[0]}%", f"{zipf_y[1]}%", f"{zipf_y[2]}%"]])
    table.align = "l"
    print(table)
    graph([1, 2, 3], nltk_y, spacy_y, bert_y, zipf_y, labels=["NLTK", "SpaCy", "BERT", "Zipf"])

def main():
    text = read('wsj_untokenized.txt')
    sentence = " ''Cosby'' is down a full ratings point in the week of Oct. 2-8 over the same week a year ago, according to A.C. Nielsen Co. Mr. Gillespie at Viacom says the ratings are rising."
    
    # QUESTION.1 & QUESTION.2
    t1a, t2a, t3a = tokenize(text, False)
    #count(t1a, t2a, t3a)
    
    # QUESTION.3 TODOOOOO
    #t1b, t2b, t3b = tokenize(sentence, True)
    
    # QUESTION.4
    #get_statistics(t1a, t2a, t3a)
    
    # QUESTION.5
    #get_occurancies(t1a, t2a, t3a)
    
    # QUESTION.6
    #get_occurancies(t1a, t2a, t3a)
    
if __name__ == "__main__":
    main()
