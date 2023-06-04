

import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter


def my_autocorrect(input_word, V):
    input_word = input_word.lower()
    df = pd.DataFrame(columns=['Sim', 'Word'])
    if input_word not in V:
        for v in V:
            sim = 1-(textdistance.Jaccard(qval=2).distance(v,input_word))
            df.loc[len(df)] = {'Sim': sim, 'Word': v}
            # df = df.append({'Sim': sim, 'Word': v}, ignore_index=True)
    output = df.head(80)
    print(output)

if __name__ == "__main__":
    words = []

    with open('C:/Users/eduar/Documents/Pos-MachineLearning/Data/auto.txt', 'r') as f:
        words = f.readlines()
        # file_name_data = f.read()
        # file_name_data = file_name_data.lower()
        # words = re.findall('w+', file_name_data)
    # This is our vocabulary
    V = set(words)
    # print("Top ten words in the text are:{words[0:10]}")
    # print("Total Unique words are {len(V)}.")
    #
    # word_freq = {}
    # word_freq = Counter(words)
    # print(word_freq.most_common()[0:10])
    #
    # probs = {}
    # Total = sum(word_freq.values())
    # for k in word_freq.keys():
    #     probs[k] = word_freq[k] / Total

    my_autocorrect("mega xburgue", V)
