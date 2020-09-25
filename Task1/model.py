import os
import re
import math
import time
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

np.random.seed(5)
stop_words = stopwords.words('english')

def process(text):
    text = re.sub(r'\'m', ' am', text)
    text = re.sub(r'\'re', ' are', text)
    text = re.sub(r'\'ll', ' will', text)
    text = re.sub(r'\'s', ' is', text)
    text = re.sub(r'\'t', ' not', text)
    text = re.sub(r'\'ve', ' have', text)
    text = re.sub(r'\'d', ' would', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = text.lower()
    text = text.strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = (' ').join(words)

    return text

def load_data(path):
    print("Loading data...")
    raw_data = pd.read_csv(path, sep='\t')
    Phrases = raw_data['Phrase']
    Sentiment = raw_data['Sentiment']
    text_list = []
    labels_list = []
    for idx in range(len(Phrases)):
        text = Phrases[idx]
        text = process(text)
        if text == "":
            continue
        text_list.append(text)
        labels_list.append(Sentiment[idx])

    print("Loading data successfully...")
    return text_list, labels_list

def word2vec(text_list, labels_list, ngram):
    print("Get word vecor by ngram and split data...")
    vectorizer = CountVectorizer(ngram_range=(ngram, ngram))
    # vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(text_list)
    data = data.toarray()

    label = np.zeros((len(labels_list), 5))
    for i in range(len(labels_list)):
        label[i][labels_list[i]] = 1

    train_data, dev_data, train_label, dev_label = train_test_split(data, label, test_size=0.2, random_state=0)

    return train_data, train_label, dev_data, dev_label

def Cross_Entropy_func(W, batch_data, batch_label):
    n_samples = batch_data.shape[0]
    up_num = np.exp(batch_data.dot(W))
    down_num = np.dot(up_num, np.ones((batch_label.shape[1], batch_label.shape[1])))
    loss = up_num / down_num
    loss = np.log(loss)
    loss = batch_label * loss
    loss = -1/n_samples * loss.sum()
    gradient = np.transpose(batch_data).dot(batch_label - up_num/down_num)
    gradient = gradient / n_samples
    return loss, gradient

def MSE_func(W, batch_data, batch_label):
    n_samples = batch_data.shape[0]
    up_num = np.exp(batch_data.dot(W))
    down_num = np.dot(up_num, np.ones((batch_label.shape[1], batch_label.shape[1])))
    batch_pred = up_num / down_num
    loss = np.power(batch_label - batch_pred, 2)
    loss = 1/n_samples * loss.sum()
    gradient = 2 / n_samples * (batch_label - batch_pred)
    gradient = np.transpose(batch_data).dot(batch_pred * (gradient - (gradient * batch_pred).sum()))

    return loss, gradient

def train(train_data, train_label, dev_data, dev_label, batch_size, learning_rate, epoches):
    print("Training...")
    train_num = train_data.shape[0]
    dev_num = dev_data.shape[0]
    in_size = train_data.shape[1]
    out_size = train_label.shape[1]
    W = np.random.random((in_size+1, out_size)) / 10
    # W = np.zeros((in_size, out_size))
    bacth_num = math.floor(train_num / batch_size)

    global_steps = 0
    best_epoch = 0
    smallest_loss = 0.0
    best_val_accuary = 0.0
    # train_data = np.insert(train_data, in_size, values=np.ones(train_num), axis=1)
    dev_data = np.insert(dev_data, in_size, values=np.ones(dev_num), axis=1)
    start = time.clock()
    for epoch in range(epoches):
        state = np.random.get_state()
        np.random.shuffle(train_data)
        np.random.set_state(state)
        np.random.shuffle(train_label)
        for batch_idx in range(bacth_num):
            global_steps += 1
            batch_data = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
            batch_data = np.insert(batch_data, in_size, values=np.ones(batch_size), axis=1)
            batch_label = train_label[batch_idx*batch_size:(batch_idx+1)*batch_size, :]

            # up_num = np.exp(batch_data.dot(W))
            # down_num = np.dot(up_num, np.ones((out_size, out_size)))
            # gradient = np.transpose(batch_data).dot(batch_label - up_num/down_num)
            # gradient = gradient / batch_size
            # loss, gradient = Cross_Entropy_func(W, batch_data, batch_label)
            loss, gradient = MSE_func(W, batch_data, batch_label)
            W = W + learning_rate * gradient

            if global_steps % 200 == 0:
                if math.isnan(loss.item()):
                    print(W)
                accu = accuary(W, batch_data, batch_label)
                print("epoch: {}, loss: {:.4f}, accuary: {:.4f}".format(epoch+1, loss, accu))

        val_accuary = evaluation(W, dev_data, dev_label, batch_size=64)
        if best_val_accuary < val_accuary:
            best_epoch = epoch + 1
            smallest_loss, _ = MSE_func(W, dev_data, dev_label)
            best_val_accuary = val_accuary
        print("="*60)
        print("Best_epoch: {}, Smallest_loss: {:.4f}, Best_val_accuary: {:.4f}".format(best_epoch, smallest_loss, best_val_accuary))
        print("="*60)

    end = time.clock()
    train_time = end - start
    print("The time of training the model is {:.3f} seconds.".format(train_time))

def accuary(W, batch_data, batch_label):
    c = batch_label.shape[1]
    up_num = np.dot(batch_data, W)
    up_num = np.exp(up_num)
    down_num = np.dot(up_num, np.ones((c, c)))
    batch_pred = up_num / down_num
    batch_pred = np.argmax(batch_pred, axis=1)
    true_label = np.argmax(batch_label, axis=1)
    correct_num = np.sum(batch_pred == true_label)

    accu = correct_num.item() / batch_data.shape[0]

    return accu

def evaluation(W, dev_data, dev_label, batch_size):
    # in_size = dev_data.shape[1]
    # dev_num = dev_data.shape[0]
    # dev_data = np.insert(dev_data, in_size, values=np.ones(dev_num), axis=1)
    dev_num = dev_data.shape[0]
    batch_num = math.floor(dev_num / batch_size)

    accu = 0.0
    for batch_idx in range(batch_num):
        if batch_idx < batch_num-1:
            batch_data = dev_data[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
            batch_label = dev_label[batch_idx*batch_size:(batch_idx+1)*batch_size, :]

            accu += accuary(W, batch_data, batch_label)

        else:
            batch_data = dev_data[batch_idx * batch_size:dev_num, :]
            batch_label = dev_label[batch_idx * batch_size:dev_num, :]

            accu += accuary(W, batch_data, batch_label)

    accu = accu / batch_num
    return accu

path = './data/train.tsv'
text_list, label_list = load_data(path)
train_data, train_label, dev_data, dev_label = word2vec(text_list, label_list, ngram=1)

print("# of examples to train: %d" % train_data.shape[0])
print("# of examples to validate: %d" % dev_data.shape[0])

train(train_data, train_label, dev_data, dev_label, batch_size=64, learning_rate=2, epoches=30)