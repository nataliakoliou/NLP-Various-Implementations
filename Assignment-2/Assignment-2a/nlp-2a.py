import gensim.downloader as api
from prettytable import PrettyTable

def load_embeddings():
    return api.load("word2vec-google-news-300"), api.load("glove-wiki-gigaword-300")

def get_similar_words(n, models, words, pos, neg):
    sims = []
    for model_name, model in models.items():
        temp = []
        pt = PrettyTable(field_names=[f"\033[1m{word}\033[0m" for word in words])
        for word in words:
            temp.extend([f"{s[0]}: {s[1]:.4f}" for s in model.most_similar(positive=pos+[word], negative=neg, topn=n)]) if word in model else temp.extend(["N/A"] * n)
        for i in range(n):
            pt.add_row([temp[i + j*n] for j in range(len(words))])
        sims.append([elem.split(':')[0].strip() if ":" in elem else "N/A" for elem in temp])
        print('\033[1m' + f"\n{model_name} Model:" + '\033[0m')
        pt.align = 'l'
        print(pt)
    return sims

def get_common_words(n, words, sims):
    coms = []
    pt = PrettyTable()
    for i in range(0, n*len(words), n):  # n*len(words) = 40
        if 'N/A' in sims[0][i:i+n] and 'N/A' in sims[1][i:i+n]:
            coms.append([])
        else:
            coms.append([word for word in set(sims[0][i:i+n]).intersection(set(sims[1][i:i+n]))])
    coms = [sublist + [""] * (max(map(len, coms)) - len(sublist)) for sublist in coms]
    for i in range(0, n*len(words), n): 
        pt.add_column(f"\033[1m{words[i // n]}\033[0m", coms[i // n]) # where current word = words[i // n]
    print('\033[1m' + "Common words in both Models:" + '\033[0m')
    pt.align = 'l'
    print(pt)

w2v_model, glove_model = load_embeddings() # loads the pre-trained word embeddings for word2vec and GloVe

sims = get_similar_words(10, {'Word2vec': w2v_model, 'GloVe': glove_model}, ['car', 'jaguar', 'Jaguar', 'facebook'], [], [])
get_common_words(10, ['car', 'jaguar', 'Jaguar', 'facebook'], sims)

sims = get_similar_words(10, {'Word2vec': w2v_model, 'GloVe': glove_model}, ['country', 'crying', 'Rachmaninoff', 'espresso'], [], [])
get_common_words(10, ['country', 'crying', 'Rachmaninoff', 'espresso'], sims)

sims = get_similar_words(10, {'Word2vec': w2v_model, 'GloVe': glove_model}, ['student'], [], [])
sims = get_similar_words(10, {'Word2vec': w2v_model, 'GloVe': glove_model}, ['student'], [], ['university'])
sims = get_similar_words(10, {'Word2vec': w2v_model, 'GloVe': glove_model}, ['student'], [], ['elementary','middle','high'])
