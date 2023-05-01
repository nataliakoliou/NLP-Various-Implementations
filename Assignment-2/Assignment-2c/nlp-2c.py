import time
import torch
import random
import pandas as pd
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from prettytable import PrettyTable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def set_device(primary, secondary):
    return torch.device(primary if torch.cuda.is_available() else secondary) # device used to perform the computations for the machine learning model

def load_dataset(path, features, label, percent, mode):
    data = pd.read_csv(path)
    data = pd.concat([data.iloc[:1], data.iloc[1:].sample(frac=1)], ignore_index=True)  # shuffle all rows except the first one
    if mode == 'start':
        end_index = int(len(data) * (percent / 100))
        data = data.iloc[:end_index]
    elif mode == 'end':
        start_index = int(len(data) * ((100 - percent) / 100))
        data = pd.concat([data.iloc[0:0], data.iloc[start_index:]], ignore_index=True)
    text = data[features].astype(str).agg(' '.join, axis=1)
    return [(data[label][i], text[i]) for i in range(len(data))]

def load_embeddings(path, vocab, dimension):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    embeddings = torch.zeros(len(vocab), dimension)
    for line in lines:
        word, vec = line.strip().split(' ', 1)
        if word in vocab:
            embeddings[vocab[word]] = torch.tensor([float(x) for x in vec.split()])
    return embeddings

def collate_batch(batch, max_words):
    Y, X = list(zip(*batch))
    Y = torch.tensor(Y) - 1  # target names in range [0,1,2,3] instead of [1,2,3,4]
    X = [vocab(tokenizer(text)) for text in X] # type(X): list of lists
    X = [tokens+([vocab['<PAD>']]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X]  # brings all samples to MAX_WORDS length - shorter texts are padded with <PAD> sequences, longer texts are truncated
    return torch.tensor(X, dtype=torch.int32).to(device), Y.to(device)

def generate_loader(dataset, max_words, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: collate_batch(b, max_words))

def tokenize(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

def build_vocab(datasets, min_freq, padded, unknown):
    vocab = build_vocab_from_iterator(tokenize(datasets), min_freq=min_freq, specials=[padded, unknown])
    vocab.set_default_index(vocab[unknown])
    return vocab
  
class RNN_model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional, output_dim, none, freeze):
        super(RNN_model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden_size = hidden_dim * get_directions(bidirectional)
        self.linear = nn.Linear(hidden_dim * get_directions(bidirectional), output_dim)
    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings)
        output_concat = torch.cat([output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]], dim=2) # concatenates outputs
        logits = self.linear(output_concat[:, -1, :]) # the last output of the concatenated RNN is used for sequence classification
        probs = F.softmax(logits, dim=1)
        return probs
    
class LSTM_model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional, output_dim, none, freeze):
        super(LSTM_model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden_size = hidden_dim * get_directions(bidirectional)
        self.linear = nn.Linear(hidden_dim * get_directions(bidirectional), output_dim)
    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, (hidden, cell) = self.lstm(embeddings)
        output_concat = torch.cat([output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]], dim=2) # concatenates outputs
        logits = self.linear(output_concat[:, -1, :]) # the last output of the concatenated LSTM is used for sequence classification
        probs = F.softmax(logits, dim=1)
        return probs
    
class pretrained_RNN_model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional, output_dim, embeddings, freeze):
        super(pretrained_RNN_model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.embedding_layer.weight.data.copy_(embeddings)
        self.embedding_layer.weight.requires_grad = freeze  # freezes the weights of the embedding layer
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden_size = hidden_dim * get_directions(bidirectional)
        self.linear = nn.Linear(hidden_dim * get_directions(bidirectional), output_dim)
    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings)
        output_concat = torch.cat([output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]], dim=2) # concatenates outputs
        logits = self.linear(output_concat[:, -1, :]) # the last output of the concatenated RNN is used for sequence classification
        probs = F.softmax(logits, dim=1)
        return probs

class pretrained_LSTM_model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional, output_dim, embeddings, freeze):
        super(pretrained_LSTM_model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.embedding_layer.weight.data.copy_(embeddings)
        self.embedding_layer.weight.requires_grad = freeze  # freezes the weights of the embedding layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden_size = hidden_dim * get_directions(bidirectional)
        self.linear = nn.Linear(hidden_dim * get_directions(bidirectional), output_dim)
    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, (hidden, cell) = self.lstm(embeddings)
        output_concat = torch.cat([output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]], dim=2) # concatenates outputs
        logits = self.linear(output_concat[:, -1, :]) # the last output of the concatenated LSTM is used for sequence classification
        probs = F.softmax(logits, dim=1)
        return probs

def setup_model(device, model, classes, vocab, embedding_dim, hidden_dim, num_layers, bidirectional, learning_rate, embeddings, freeze):
    classifier = model(len(vocab), embedding_dim, hidden_dim, num_layers, bidirectional, len(classes), embeddings, freeze).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for param in classifier.parameters() if param.requires_grad == True],lr=learning_rate)
    return classifier, loss_fn, optimizer

def train_model(classifier, loss_fn, optimizer, train_loader, epochs):
    times = []
    for i in range(1, epochs+1):
        classifier.train()
        print('\033[1mEpoch\033[0m:',i)
        losses = []
        start_time = time.time()
        for X, Y in tqdm(train_loader):
            Y_preds = classifier(X)
            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        print("\033[1mTrain Loss\033[0m: {:.3f}\n".format(torch.tensor(losses).mean()))
    return sum(times)/len(times)

def evaluate_model(classes, classifier, loss_fn, test_loader, test_data):
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
        Y_actual, Y_preds = Y_actual.detach().cpu().numpy(), Y_preds.detach().cpu().numpy()
        misclass_data = detect_misclassification(test_data, Y_actual, Y_preds)
        print("\033[1mTest Accuracy\033[0m: {:.3f}\n".format(accuracy_score(Y_actual, Y_preds)))
        print("\033[1mClassification Report:\033[0m\n", classification_report(Y_actual, Y_preds, target_names=classes))
        print("\033[1mConfusion Matrix:\033[0m\n", confusion_matrix(Y_actual, Y_preds))
    return torch.tensor(losses).mean(), Y_actual, Y_preds, misclass_data      

def detect_misclassification(test_data, Y_actual, Y_preds):
    misclass_data = defaultdict(list)
    for i in range(len(Y_actual)):
        true_label = Y_actual[i]
        predicted_label = Y_preds[i]
        if true_label != predicted_label:
            text = test_data["features"][i]
            misclass_data[true_label].append((text, predicted_label))
    return misclass_data

def analyze_results(classes, models, misclassified):
    common_misclass_data = defaultdict(list)
    for true_label in misclassified[0].keys():
        for text, label in misclassified[0][true_label]:
            labels = [label] + [next((l for t, l in model[true_label] if t == text), '') for model in misclassified[1:]]
            common_misclass_data[true_label].append((text, labels)) if all(labels) else None
    count_times(common_misclass_data, classes)
    get_top_pair(common_misclass_data, classes)
    get_random_text(models, common_misclass_data, classes)

def count_times(common_misclass_data, classes):
    misclass_counts = {true_label: len(misclass_tuples) for true_label, misclass_tuples in common_misclass_data.items()}
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["True Label", "Misclassified Texts"]])
    [pt.add_row([to_category(true_label, classes), count]) for true_label, count in misclass_counts.items()]
    print("\033[1mCommon Misclassified Texts per Class:\033[0m")
    print(pt)

def get_top_pair(common_misclass_data, classes):
    misclass_freqs = defaultdict(int)
    for true_label, values in common_misclass_data.items():
        for text, pred_labels in values:
            for pl in pred_labels:
                misclass_freqs[(true_label, pl)] += 1
    max_tuple, max_count = max(misclass_freqs.items(), key=lambda x: x[1])
    sorted_tuples = sorted(misclass_freqs.items(), key=lambda x: x[1], reverse=True)
    print(f"\n\033[1mMost common Misclassification Pair:\033[0m ({to_category(max_tuple[0], classes)}, {to_category(max_tuple[1], classes)})")
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["True Label", "Predicted Label", "Frequency"]])
    [pt.add_row([to_category(tup[0], classes), to_category(tup[1], classes), count]) for tup, count in sorted_tuples]
    print(pt)

def get_random_text(models, common_misclass_data, classes):
    rand_true_label = random.choice(list(common_misclass_data.keys()))
    rand_misclass_tuple = random.choice(common_misclass_data[rand_true_label])
    print("\n\033[1m" + "Random Text: " + "\033[0m" + rand_misclass_tuple[0] + "\033[1m" + "\nTrue Label: " + "\033[0m" + to_category(rand_true_label, classes))
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Model", "Prediction"]])
    [pt.add_row([model, to_category(rand_misclass_tuple[1][idx], classes)]) for idx, model in enumerate(models)]
    print(pt)

def visualize(models, accuracies, parameters, time_costs):
    pt = PrettyTable(field_names=[f"\033[1m{field}\033[0m" for field in ["Model", "Accuracy", "Parameters", "Time Cost"]])
    [pt.add_row([model, round(accuracies[i], 4), parameters[i], round(time_costs[i], 4)]) for i, model in enumerate(models)]
    print(pt)

def replace_labels(dataset, categorical, numerical):
    mapping = {categorical[0]: numerical[0], categorical[1]: numerical[1]}
    return [(mapping[label], text) for label, text in dataset]

def to_dict(tuples_list):
    return {'features': [d[1] for d in tuples_list], 'labels': [d[0] for d in tuples_list]}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_directions(bidirectional):
    return 2 if bidirectional else 1

def to_category(label, classes):
    return classes[label]

##########################################################################################################################################################################################
device = set_device("cuda","cpu")
tokenizer = get_tokenizer("basic_english")

models = ["1Uni-RNN", "1Bi-RNN", "2Bi-RNN", "1Uni-LSTM", "1Bi-LSTM", "2Bi-LSTM"]; classes = ["World", "Sports", "Business", "Sci/Tech"]; accuracies = []; parameters = []; time_costs = []
MIN_FREQ = 10 ; MAX_WORDS = 25; EPOCHS = 15; LEARNING_RATE = 1e-3; BATCH_SIZE = 1024; EMBEDDING_DIM = 100; HIDDEN_DIM = 64; PADDED = "<PAD>"; UNKNOWN = "<UNK>"

train_dataset, test_dataset = load_dataset("train.csv", ["Title","Description"], "Class Index", 100, "start"), load_dataset("test.csv", ["Title","Description"], "Class Index", 100, "start")
train_loader, test_loader = generate_loader(train_dataset, MAX_WORDS, BATCH_SIZE, True), generate_loader(test_dataset, MAX_WORDS, BATCH_SIZE, False)
vocab = build_vocab([train_dataset, test_dataset], MIN_FREQ, PADDED, UNKNOWN)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

visualize(models, accuracies, parameters, time_costs)

##########################################################################################################################################################################################

analyze_results(classes, models, [misclass_data_1UniRNN, misclass_data_1BiRNN, misclass_data_2BiRNN, misclass_data_1UniLSTM, misclass_data_1BiLSTM, misclass_data_2BiLSTM])

##########################################################################################################################################################################################

MAX_WORDS = 50
train_loader, test_loader = generate_loader(train_dataset, MAX_WORDS, BATCH_SIZE, True), generate_loader(test_dataset, MAX_WORDS, BATCH_SIZE, False)
accuracies = []; parameters = []; time_costs = []

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

visualize(models, accuracies, parameters, time_costs)

##########################################################################################################################################################################################

MAX_WORDS = 25
models = ["1Uni-preRNN", "1Bi-preRNN", "2Bi-preRNN", "1Uni-preLSTM", "1Bi-preLSTM", "2Bi-preLSTM"]; accuracies = []; parameters = []; time_costs = []

embeddings = load_embeddings("glove.6B.100d.txt", vocab, EMBEDDING_DIM)
train_loader, test_loader = generate_loader(train_dataset, MAX_WORDS, BATCH_SIZE, True), generate_loader(test_dataset, MAX_WORDS, BATCH_SIZE, False)

classifier, loss_fn, optimizer = setup_model(device, pretrained_RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, embeddings, False)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, embeddings, False)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, embeddings, False)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, embeddings, False)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, embeddings, False)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, embeddings, False)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

visualize(models, accuracies, parameters, time_costs)

##########################################################################################################################################################################################

accuracies = []; parameters = []; time_costs = []

classifier, loss_fn, optimizer = setup_model(device, pretrained_RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, embeddings, True)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, embeddings, True)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, embeddings, True)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, embeddings, True)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, embeddings, True)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, pretrained_LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, embeddings, True)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

visualize(models, accuracies, parameters, time_costs)

##########################################################################################################################################################################################

models = ["1Uni-RNN", "1Bi-RNN", "2Bi-RNN", "1Uni-LSTM", "1Bi-LSTM", "2Bi-LSTM"]; classes = ["Positive", "Negative"]; accuracies = []; parameters = []; time_costs = []

train_dataset, test_dataset = load_dataset("IMDB Dataset.csv", ["review"], "sentiment", 80, "start"), load_dataset("IMDB Dataset.csv", ["review"], "sentiment", 20, "end")
train_dataset, test_dataset = replace_labels(train_dataset, ["negative", "positive"], [1,2]), replace_labels(test_dataset, ["negative", "positive"], [1,2])
train_loader, test_loader = generate_loader(train_dataset, MAX_WORDS, BATCH_SIZE, True), generate_loader(test_dataset, MAX_WORDS, BATCH_SIZE, False)
vocab = build_vocab([train_dataset, test_dataset], MIN_FREQ, PADDED, UNKNOWN)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, RNN_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiRNN = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, False, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1UniLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 1, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_1BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

classifier, loss_fn, optimizer = setup_model(device, LSTM_model, classes, vocab, EMBEDDING_DIM, HIDDEN_DIM, 2, True, LEARNING_RATE, None, None)
time_cost = train_model(classifier, loss_fn, optimizer, train_loader, EPOCHS)
_, Y_actual, Y_preds, misclass_data_2BiLSTM = evaluate_model(classes, classifier, loss_fn, test_loader, to_dict(test_dataset))
accuracies.append(accuracy_score(Y_actual, Y_preds))
parameters.append(count_parameters(classifier))
time_costs.append(time_cost)

visualize(models, accuracies, parameters, time_costs)
