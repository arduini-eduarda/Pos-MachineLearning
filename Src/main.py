import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import tflearn
import json
import pickle
import re
import os
import matplotlib.pyplot as plt
import random

from nltk.corpus import stopwords
nltk.download('stopwords')

nltk.download('punkt')
from Order.Items import findOrder
# from DataProcessing.TfIdf import tfIdf

accuracy = []
loss = []


class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, api):
        self.my_monitor_api = api

    def on_sub_batch_end(self, training_state, train_index=0):
        try:
            accuracy.append(str(training_state.acc_value))
            loss.append(str(training_state.loss_value))
        except Exception as e:
            print(str(e))


monitorCallback = MonitorCallback(tflearn)


def bag_of_words(s, words):
    stemmer = SnowballStemmer("portuguese")
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    # Loading intents.json
    with open('../Data/intents.json') as intents:
        data = json.load(intents)

    stemmer = SnowballStemmer("portuguese")

    if os.path.isfile('data.pickle'):
        with open('data.pickle', 'rb') as f:
            words, labels, training, output = pickle.load(f)
    else:
        # Fetching and Feeding information--
        words = []
        labels = []
        x_docs = []
        y_docs = []

        # used to generate tfidf from words
        # tfIdf(data['intents'])

        for intent in data['intents']:
            words_to_cloud = []

            for pattern in intent['patterns']:

                pattern = re.sub(r"[^\w\s]|_", "", pattern)

                wrds = nltk.word_tokenize(pattern.lower(), language='portuguese')

                all_stop_words = stopwords.words("portuguese")

                list_to_remove = ["mais", "sem", "um", "uma", "só"]

                real_stop_words = [word for word in all_stop_words if not word in list_to_remove]

                tokens_without_sw = [word for word in wrds if not word in real_stop_words]

                words.extend(tokens_without_sw)

                words_to_cloud.extend(tokens_without_sw)

                x_docs.append(tokens_without_sw)
                y_docs.append(intent['tag'])

                if intent['tag'] not in labels:
                    labels.append(intent['tag'])


        words = [stemmer.stem(w) for w in words if w not in "?"]
        words = sorted(list(set(words)))
        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        # One hot encoding, Converting the words to numerals
        for x, doc in enumerate(x_docs):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(y_docs[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = np.array(training)
        output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

    model = cnn_net(output, training)

    if os.path.isfile('model.tflearn'):
        model.load("model.tflearn")
    else:
        model.fit(training, output, n_epoch=100, batch_size=6, show_metric=True, callbacks=monitorCallback)

    # used to generate model graphic
    # saveModelData()

    model.save('model.tflearn')

    print("The bot is ready to talk!!(Type 'quit' to exit)")
    print("Bot: Olá, gostaria de conhecer nosso menu ou já quer fazer o pedido?")
    while True:

        inp = input("\nYou: ")
        if inp.lower() == 'quit':
            break

        # Porbability of correct response
        results = model.predict([bag_of_words(inp, words)])

        # Picking the greatest number from probability
        results_index = np.argmax(results)

        tag = labels[results_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                if tg['tag'] == 'order':
                    findOrder(inp.lower())

                responses = tg['responses']
                print("Bot:" + random.choice(responses))
                break


def saveModelData():
    floatAccuracy = [float(x) for x in accuracy]
    floatLoss = [float(x) for x in loss]
    plt.plot(floatAccuracy, 'g', label='Acurácia')
    plt.plot(floatLoss, 'r', label='Perda')
    plt.legend(['Acurácia', 'Perda'])
    plt.savefig('accuracy.png', bbox_inches='tight')


def cnn_net(output, training):
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    return model


# initiate the conversation
if __name__ == "__main__":
    # createData()
    chat()
