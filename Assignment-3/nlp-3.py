import torch
import numpy as np
import pandas as pd
import torch.optim as optim 
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertForTokenClassification, BertTokenizerFast, RobertaForTokenClassification, RobertaTokenizerFast
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import tqdm

def load_sentences(filepath):
    sentences, tokens, pos_tags, chunk_tags, ner_tags = [], [], [], [], []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                if len(tokens) > 0: # only-case: empty line, which indicates the end of a sentence
                    sentences.append({'tokens': tokens, 'pos_tags': pos_tags, 'chunk_tags': chunk_tags, 'ner_tags': ner_tags})
                    tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
            else:
                l = line.split(' ')
                tokens.append(l[0])
                pos_tags.append(l[1])
                chunk_tags.append(l[2])
                ner_tags.append(l[3].strip('\n'))
    return sentences

def load_data(base_path):
    print('\033[1mLoading data:\033[0m')
    train_sentences = load_sentences(base_path + 'train.txt')
    test_sentences = load_sentences(base_path + 'test.txt')
    valid_sentences = load_sentences(base_path + 'valid.txt')
    example_sentences = load_sentences(base_path + 'example.txt')
    return train_sentences, test_sentences, valid_sentences, example_sentences

def create_tagset(sentences, tagtype):
    tags = [sentence[tagtype] for sentence in sentences]
    tagmap = build_vocab_from_iterator(tags)
    tagset = set([item for sublist in tags for item in sublist])
    print('Tagset size:', len(tagset))
    return tagmap, tagset

def align_label(tagmap, tokens, labels):
    word_ids = tokens.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:  # case: special or padding token
            label_ids.append(-100)
        elif word_idx != previous_word_idx: #  case: token represents a new word or entity
            try:
                label_ids.append(tagmap[labels[word_idx]])
            except:
                label_ids.append(-100)
        else: # case: consecutive tokens in the sequence belong to the same word or entity
            label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids

def encode_sentence(tagmap, tokenizer, sentence, tagtype):
    encodings = tokenizer(sentence['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
    labels = align_label(tagmap, encodings, sentence[tagtype])
    return { 'input_ids': torch.LongTensor(encodings.input_ids), 'attention_mask': torch.LongTensor(encodings.attention_mask), 'labels': torch.LongTensor(labels) }

def encode_data(tagmap, tokenizer, train_sentences, valid_sentences, test_sentences, example_sentences, tagtype):
    print('\033[1mEncoding data:\033[0m')
    train_dataset = [encode_sentence(tagmap, tokenizer, sentence, tagtype) for sentence in train_sentences]
    valid_dataset = [encode_sentence(tagmap, tokenizer, sentence, tagtype) for sentence in valid_sentences]
    test_dataset = [encode_sentence(tagmap, tokenizer, sentence, tagtype) for sentence in test_sentences]
    example_dataset = [encode_sentence(tagmap, tokenizer, sentence, tagtype) for sentence in example_sentences]
    return train_dataset, valid_dataset, test_dataset, example_dataset

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    data = [{'Total': total_params, 'Trainable': trainable_params, 'Frozen': frozen_params}]
    df = pd.DataFrame(data, index=['Parameters'])
    display(df)

def setup(model_name, tagset, update):
    if model_name == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tagset))
        model.bert.requires_grad_(update)  # updates (True) or freezes (False) the weights of the pre-trained BERT model
    elif model_name == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
        model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(tagset))
        model.roberta.requires_grad_(update)  # updates (True) or freezes (False) the weights of the pre-trained Roberta model
    return model, tokenizer

def initialize_model(model_name, tagset, device, lr, update):
    print('\033[1mInitializing the model:\033[0m')
    model, tokenizer = setup(model_name, tagset, update)
    model.to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    count_parameters(model)
    return model, tokenizer, optimizer

def create_loaders(train_dataset, valid_dataset, test_dataset, example_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    example_loader = torch.utils.data.DataLoader(example_dataset, batch_size=batch_size)
    return train_loader, valid_loader, test_loader, example_loader

def evaluate_model(tqdmn, device, model, data_loader, detect, tokenizer):
    found = False
    model.eval()
    with torch.no_grad():
        Y_actual, Y_preds = [],[]
        for i, batch in enumerate(tqdmn(data_loader)):
            batch = { k: v.to(device) for k, v in batch.items() }
            outputs = model(**batch)
            for idx, _ in enumerate(batch['labels']):
                true_values_all = batch['labels'][idx]
                true_values = true_values_all[true_values_all != -100]
                pred_values = torch.argmax(outputs[1], dim=2)[idx]
                pred_values = pred_values[true_values_all != -100]
                Y_actual.append(true_values)
                Y_preds.append(pred_values)
                found = detect_misclassification(batch, idx, tokenizer, true_values_all, true_values, pred_values) if detect and not found else found
        Y_actual = torch.cat(Y_actual).detach().cpu().numpy()
        Y_preds = torch.cat(Y_preds).detach().cpu().numpy()
    return Y_actual, Y_preds

def train_model(model, optimizer, train_loader, valid_loader, tqdmn, device, epochs, tagmap, tokenizer):
    print('\033[1mTraining the model:\033[0m')
    for epoch in tqdmn(range(epochs)):
        model.train()
        print('Epoch', epoch+1)
        for i, batch in enumerate(tqdmn(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if valid_loader:
            Y_actual, Y_preds = evaluate_model(tqdmn, device, model, valid_loader, False, tokenizer)
            display_results("valid", Y_actual, Y_preds, tagmap)
    return model

def display_results(data_type, Y_actual, Y_preds, tagmap):
    if data_type == "valid":
        print("\nValidation Accuracy: {:.3f}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nValidation Macro-Accuracy: {:.3f}".format(balanced_accuracy_score(Y_actual, Y_preds)))
    elif data_type == "test":
        print("\nTest Accuracy : {:.3f}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nTest Macro-Accuracy : {:.3f}".format(balanced_accuracy_score(Y_actual, Y_preds)))
        print("\nClassification Report:\n{}".format(classification_report(Y_actual, Y_preds, labels=tagmap(tagmap.get_itos()), target_names=tagmap.get_itos(), zero_division=0)))

def detect_misclassification(batch, idx, tokenizer, true_values_all, true_values, pred_values):
    if len(true_values) >= 10 and not torch.equal(true_values, pred_values):
        tokens = [t for i, t in enumerate(tokenizer.convert_ids_to_tokens(batch['input_ids'][idx])) if true_values_all[i] != -100]
        data = [{'Token': token, 'Missclassified': False if true_values[i] == pred_values[i] else True, 'True': true_values[i].item(), 'Predicted': pred_values[i].item()}
            for i, token in enumerate(tokens)]
        df = pd.DataFrame(data)
        display(df)
        return True

# QUESTION.1/2a #############################################################################################################################################################

EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-5
tqdmn = tqdm.notebook.tqdm
base_path = '/content/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # uses GPU if available

train_sentences, test_sentences, valid_sentences, example_sentences = load_data(base_path)
tagmap, tagset = create_tagset(train_sentences, 'ner_tags')
model, tokenizer, optimizer = initialize_model("bert", tagset, device, LR, True)
train_dataset, valid_dataset, test_dataset, example_dataset = encode_data(tagmap, tokenizer, train_sentences, valid_sentences, test_sentences, example_sentences, 'ner_tags')
train_loader, valid_loader, test_loader, example_loader = create_loaders(train_dataset, valid_dataset, test_dataset, example_dataset, BATCH_SIZE)
model = train_model(model, optimizer, train_loader, valid_loader, tqdmn, device, EPOCHS, tagmap, tokenizer)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, test_loader, True, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap)

# QUESTION.2b ###############################################################################################################################################################
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, example_loader, True, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap) #TODO: change "test" to "example" and edit def display_results!

# QUESTION.4 ################################################################################################################################################################
model, tokenizer, optimizer = initialize_model("bert", tagset, device, LR, False)
model = train_model(model, optimizer, train_loader, valid_loader, tqdmn, device, EPOCHS, tagmap, tokenizer)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, test_loader, False, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap)

# QUESTION.5 ################################################################################################################################################################
train_sentences.extend(valid_sentences)  # Concatenate the training and validation sentences
tagmap, tagset = create_tagset(train_sentences, 'ner_tags')
model, tokenizer, optimizer = initialize_model("bert", tagset, device, LR, True)
train_dataset, _, test_dataset, _ = encode_data(tagmap, tokenizer, train_sentences, [], test_sentences, [], 'ner_tags')
train_loader, _, test_loader, _ = create_loaders(train_dataset, [], test_dataset, [], BATCH_SIZE)
model = train_model(model, optimizer, train_loader, [], tqdmn, device, EPOCHS, tagmap, tokenizer)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, test_loader, False, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap)

# QUESTION.6 ################################################################################################################################################################
train_sentences, test_sentences, valid_sentences, example_sentences = load_data(base_path)
tagmap, tagset = create_tagset(train_sentences, 'pos_tags')
model, tokenizer, optimizer = initialize_model("bert", tagset, device, LR, True)
train_dataset, valid_dataset, test_dataset, example_dataset = encode_data(tagmap, tokenizer, train_sentences, valid_sentences, test_sentences, example_sentences, 'ner_tags')
train_loader, valid_loader, test_loader, example_loader = create_loaders(train_dataset, valid_dataset, test_dataset, example_dataset, BATCH_SIZE)
model = train_model(model, optimizer, train_loader, valid_loader, tqdmn, device, EPOCHS, tagmap, tokenizer)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, test_loader, True, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, example_loader, True, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap) #TODO: change "test" to "example" and edit def display_results!

# QUESTION.7 ################################################################################################################################################################
tagmap, tagset = create_tagset(train_sentences, 'chunk_tags')
model, tokenizer, optimizer = initialize_model("bert", tagset, device, LR, True)
train_dataset, valid_dataset, test_dataset, example_dataset = encode_data(tagmap, tokenizer, train_sentences, valid_sentences, test_sentences, example_sentences, 'ner_tags')
train_loader, valid_loader, test_loader, example_loader = create_loaders(train_dataset, valid_dataset, test_dataset, example_dataset, BATCH_SIZE)
model = train_model(model, optimizer, train_loader, valid_loader, tqdmn, device, EPOCHS, tagmap, tokenizer)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, test_loader, True, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, example_loader, True, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap) #TODO: change "test" to "example" and edit def display_results!

# QUESTION.8 ################################################################################################################################################################
tagmap, tagset = create_tagset(train_sentences, 'ner_tags')
model, tokenizer, optimizer = initialize_model("roberta", tagset, device, LR, True)
train_dataset, valid_dataset, test_dataset, example_dataset = encode_data(tagmap, tokenizer, train_sentences, valid_sentences, test_sentences, example_sentences, 'ner_tags')
train_loader, valid_loader, test_loader, example_loader = create_loaders(train_dataset, valid_dataset, test_dataset, example_dataset, BATCH_SIZE)
model = train_model(model, optimizer, train_loader, valid_loader, tqdmn, device, EPOCHS, tagmap, tokenizer)
Y_actual, Y_preds = evaluate_model(tqdmn, device, model, test_loader, False, tokenizer)
display_results("test", Y_actual, Y_preds, tagmap)
