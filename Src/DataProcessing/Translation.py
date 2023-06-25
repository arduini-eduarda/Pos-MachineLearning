from google_trans_new import google_translator
import pandas as pd

def my_autocorrect():
    translator = google_translator()

    df = pd.read_csv("../../Data/menustat_2021_dataset.csv")

    df["food_category"] = df["food_category"].apply(lambda row: translator.translate(row, lang_tgt="pt"))

if __name__ == "__main__":
    my_autocorrect()

