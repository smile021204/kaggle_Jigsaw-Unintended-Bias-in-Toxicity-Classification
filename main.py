import numpy as np
import pandas as pd
import os
import time
import gc
import random
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

def check_interactive():
    return 'SHLVL' not in os.environ

def set_random_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed()

FASTTEXT_EMBEDDING_PATH = 'crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = 'glove.840B.300d.txt'
NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_UNITS = 4 * LSTM_UNITS
MAX_SEQUENCE_LENGTH = 220

def extract_word_vectors(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_word_embeddings(path):
    with open(path) as f:
        return dict(extract_word_vectors(*line.strip().split(' ')) for line in f)

def create_embedding_matrix(word_index, path):
    embedding_index = load_word_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []

    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def train_nn_model(model, train, test, loss_fn, output_dim, lr=0.001,
                   batch_size=512, n_epochs=4, enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

    for epoch in range(n_epochs):
        start_time = time.time()

        scheduler.step()

        model.train()
        avg_loss = 0.

        for data in train_loader:
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        test_preds = np.zeros((len(test), output_dim))

        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid_activation(model(*x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i + 1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1}/{n_epochs} \t loss={avg_loss:.4f} \t time={elapsed_time:.2f}s')

    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
    else:
        test_preds = all_test_preds[-1]

    return test_preds

class Dropout2D(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(Dropout2D, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x

class TextClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(TextClassifier, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = Dropout2D(0.3)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(DENSE_UNITS, DENSE_UNITS)
        self.linear2 = nn.Linear(DENSE_UNITS, DENSE_UNITS)

        self.linear_out = nn.Linear(DENSE_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_UNITS, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        avg_pool = torch.mean(h_lstm2, 1)
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out

def clean_text(data):
    punctuation = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€×™√²—–&'

    def remove_special_chars(text, punctuation):
        for p in punctuation:
            text = text.replace(p, ' ')
        return text

    return data.astype(str).apply(lambda x: remove_special_chars(x, punctuation))

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = clean_text(train_data['comment_text'])
y_train = np.where(train_data['target'] >= 0.5, 1, 0)
y_aux_train = train_data[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = clean_text(test_data['comment_text'])

max_features = None

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

max_features = max_features or len(tokenizer.word_index) + 1
max_features

fasttext_matrix, unknown_words_fasttext = create_embedding_matrix(tokenizer.word_index, FASTTEXT_EMBEDDING_PATH)
print('Unknown words (FastText): ', len(unknown_words_fasttext))

glove_matrix, unknown_words_glove = create_embedding_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
print('Unknown words (GloVe): ', len(unknown_words_glove))

embedding_matrix = np.concatenate([fasttext_matrix, glove_matrix], axis=-1)
embedding_matrix.shape

del fasttext_matrix
del glove_matrix
gc.collect()

x_train_torch = torch.tensor(x_train, dtype=torch.long)
x_test_torch = torch.tensor(x_test, dtype=torch.long)
y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32)

train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
test_dataset = data.TensorDataset(x_test_torch)

all_test_preds = []

for model_idx in range(NUM_MODELS):
    print('Model ', model_idx)
    set_random_seed(1234 + model_idx)

    model = TextClassifier(embedding_matrix, y_aux_train.shape[-1])

    test_preds = train_nn_model(model, train_dataset, test_dataset, output_dim=y_train_torch.shape[-1],
                                loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
    all_test_preds.append(test_preds)
    print()

result = pd.DataFrame.from_dict({
    'id': test_data['id'],
    'prediction': np.mean(all_test_preds, axis=0)[:, 0]
})

result.to_csv('result.csv', index=False)
print("Done")
