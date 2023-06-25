import pandas as pd
import textdistance


def correct(input_word, V):
    input_word = input_word.lower()
    df = pd.DataFrame(columns=['Sim', 'Word'])
    if input_word not in V:
        for v in V:
            sim = 1-(textdistance.Jaccard(qval=2).distance(v,input_word))
            df.loc[len(df)] = {'Sim': sim, 'Word': v}
    output = df.head(80)
    print(output)

# if __name__ == "__main__":
#     words = []
#
#     with open('/Data/auto.txt', 'r') as f:
#         words = f.readlines()
#     V = set(words)
#
#     my_autocorrect("mega xburgue", V)
