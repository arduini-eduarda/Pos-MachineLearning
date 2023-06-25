import spacy
nlp = spacy.load("pt_core_news_md")


def my_autocorrect():
    while True:
        print("Enter two space-separated words")
        words = input()

        tokens = nlp(words)

        for token in tokens:
            print(token.text, token.has_vector, token.vector_norm, token.is_oov)

        token1, token2 = tokens[0], tokens[1]

        print("Similarity:", token1.similarity(token2))


