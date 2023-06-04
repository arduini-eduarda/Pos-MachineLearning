from google_trans_new import google_translator
import numpy as np
import pandas as pd

def my_autocorrect():
    translator = google_translator()

    df = pd.read_csv("./../Data/menustat_2021_dataset.csv")

    df["food_category"] = df["food_category"].apply(lambda row: translator.translate(row, lang_tgt="pt"))

if __name__ == "__main__":
    my_autocorrect()


# sentence = "Tanzania ni nchi inayoongoza kwa utalii barani afrika"
# translate_text = translator.translate(sentence, lang_tgt='pt')
#
# print(translate_text)