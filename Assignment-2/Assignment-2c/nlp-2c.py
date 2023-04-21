import torch
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torch import nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def set_device(primary, secondary):
    return torch.device(primary if torch.cuda.is_available() else secondary) # device used to perform the computations for the machine learning model

def load_dataset(path):
    data = pd.read_csv(path)
    return [(label, data['Title'][i] + ' ' + data['Description'][i]) for i, label in enumerate(data['Class Index'])]

def collate_batch(batch):
    Y, X = list(zip(*batch))
    Y = torch.tensor(Y) - 1  # target names in range [0,1,2,3] instead of [1,2,3,4]
    X = [vocab(tokenizer(text)) for text in X] # type(X): list of lists
    X = [tokens+([vocab['<PAD>']]* (MAX_WORDS-len(tokens))) if len(tokens)<MAX_WORDS else tokens[:MAX_WORDS] for tokens in X]  # brings all samples to MAX_WORDS length - shorter texts are padded with <PAD> sequences, longer texts are truncated
    return torch.tensor(X, dtype=torch.int32).to(device), Y.to(device) 

def generate_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)

def tokenize(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

def build_vocab(datasets, min_freq, padded, unknown):
    vocab = build_vocab_from_iterator(tokenize(datasets), min_freq=min_freq, specials=[padded, unknown])
    vocab.set_default_index(vocab[unknown])
    return vocab

class model(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim, output_dim):  # defines the constructor of the nn model
        super(model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings)
        logits = self.linear(output[:,-1])  # The last output of RNN is used for sequence classification
        probs = F.softmax(logits, dim=1)
        return probs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_model(device, classes, train_dataset, test_dataset, min_freq, padded, unknown, embedding_dim, hidden_dim, learning_rate):
    vocab = build_vocab([train_dataset, test_dataset], min_freq, padded, unknown)
    classifier = model(len(vocab), embedding_dim, hidden_dim, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for param in classifier.parameters() if param.requires_grad == True],lr=learning_rate)
    return classifier, loss_fn, optimizer

def train_model(classifier, loss_fn, optimizer, train_loader, epochs):
    for i in range(1, epochs+1):
        classifier.train()
        print('Epoch:',i)
        losses = []
        for X, Y in tqdm(train_loader):
            Y_preds = classifier(X)
            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))

def evaluate_model(classifier, loss_fn, test_loader):
    classifier.eval()
    with torch.no_grad():  # during evaluation we don't update the model's parameters
        Y_actual, Y_preds, losses = [],[],[]
        for X, Y in test_loader:
            preds = classifier(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())
            Y_actual.append(Y)
            Y_preds.append(preds.argmax(dim=-1))
        Y_actual, Y_preds = torch.cat(Y_actual), torch.cat(Y_preds)
    return torch.tensor(losses).mean(), Y_actual.detach().cpu().numpy(), Y_preds.detach().cpu().numpy()      # returns mean loss, actual labels and predicted labels 

device = set_device("cuda","cpu")
tokenizer = get_tokenizer("basic_english")
classes = ["World", "Sports", "Business", "Sci/Tech"]
MIN_FREQ = 10 ; MAX_WORDS = 25; EPOCHS = 15; LEARNING_RATE = 1e-3; BATCH_SIZE = 1024; EMBEDDING_DIM = 100; HIDDEN_DIM = 64; PADDED = "<PAD>"; UNKNOWN = "<UNK>"

train_dataset, test_dataset = load_dataset("C:/Users/natalia/pyproj/nlp-proj/assignment-2c/train.csv"), load_dataset("C:/Users/natalia/pyproj/nlp-proj/assignment-2c/test.csv")
train_loader, test_loader = generate_loader(train_dataset, BATCH_SIZE, True), generate_loader(test_dataset, BATCH_SIZE, False)

classifier, loss_fn, optimizer = setup_model(device, classes, train_dataset, test_dataset, MIN_FREQ, PADDED, UNKNOWN, EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE)

"""
print('\nModel:')
print(classifier)
print('Total parameters: ',count_parameters(classifier))
print('\n\n')
"""

train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds = evaluate_model(classifier, loss_fn, test_loader)

#"""
print("\nTest Accuracy : {:.3f}".format(accuracy_score(Y_actual, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_actual, Y_preds, target_names=classes))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_actual, Y_preds))
#"""
