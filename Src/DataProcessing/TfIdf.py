import heapq
import nltk
import numpy as np

def tfIdf(intents):
    dataset = []

    for intent in intents:
        dataset.append(' '.join(intent["patterns"]))

    word2count = {}
    for data in dataset:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    print(word2count)

    freq_words = heapq.nlargest(50, word2count, key=word2count.get)
    print(freq_words)

    word_idfs = {}
    for word in freq_words:
        doc_count = 0
        for data in dataset:
            if word in nltk.word_tokenize(data):
                doc_count += 1
        word_idfs[word] = np.log((len(dataset) / doc_count) + 1)

    print(word_idfs)
    # calculando a frequencia de cada palavra nos documentos

    tf_matrix = {}

    for word in freq_words:
        doc_tf = []
        for data in dataset:
            frequency = 0
            for w in nltk.word_tokenize(data):
                if w == word:
                    frequency += 1
            tf_word = frequency / len(nltk.word_tokenize(data))
            doc_tf.append(tf_word)
        tf_matrix[word] = doc_tf

    print(tf_matrix)

    # TF-IDF cálculo


    tfidf_matrix = []

    for word in tf_matrix.keys():
        tfidf = []
        for value in tf_matrix[word]:
            score = value * word_idfs[word]
            tfidf.append(score)
        tfidf_matrix.append(tfidf)

    print(tfidf_matrix)

    max_word = ""
    i = 0
    max = 0
    for tfidf in tfidf_matrix:
        j = 0
        for h in tfidf:
            if tfidf[0] > max:
                max = tfidf[0]
                max_word = list(tf_matrix.keys())[i]
                j = j + 1
        i = i + 1
    print("{}: {}".format(max_word, max))

    max_word = ""
    i = 0
    max = 0
    for tfidf in tfidf_matrix:
        j = 0
        for h in tfidf:
            if tfidf[1] > max:
                max = tfidf[1]
                max_word = list(tf_matrix.keys())[i]
                j = j + 1
        i = i + 1
    print("{}: {}".format(max_word, max))


    max_word = ""
    i = 0
    max = 0
    for tfidf in tfidf_matrix:
        j = 0
        for h in tfidf:
            if tfidf[2] > max:
                max = tfidf[2]
                max_word = list(tf_matrix.keys())[i]
                j = j + 1
        i = i + 1
    print("{}: {}".format(max_word, max))
    max_word = ""
    i = 0
    max = 0


    for tfidf in tfidf_matrix:
        j = 0
        for h in tfidf:
            if tfidf[3] > max:
                max = tfidf[3]
                max_word = list(tf_matrix.keys())[i]
                j = j + 1
        i = i + 1
    print("{}: {}".format(max_word, max))


    max_word = ""
    i = 0
    max = 0
    for tfidf in tfidf_matrix:
        j = 0
        for h in tfidf:
            if tfidf[4] > max :
                max = tfidf[4]
                max_word = list(tf_matrix.keys())[i]
                j = j + 1
        i = i + 1
    print("{}: {}".format(max_word, max))
    max_word = ""
    i = 0
    max = 0
    for tfidf in tfidf_matrix:
        j = 0
        for h in tfidf:
            if tfidf[5] > max:
                max = tfidf[5]
                max_word = list(tf_matrix.keys())[i]
                j = j + 1
        i = i + 1
    print("{}: {}".format(max_word, max))
    max_word = ""
    i = 0
    max = 0
    for tfidf in tfidf_matrix:
        j = 0
        for h in tfidf:
            if tfidf[6] > max:
                max = tfidf[6]
                max_word = list(tf_matrix.keys())[i]
                j = j + 1
        i = i + 1
    print("{}: {}".format(max_word, max))
