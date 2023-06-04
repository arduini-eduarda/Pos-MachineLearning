import spacy
import tflearn
from spacy.matcher import Matcher


def findOrder(text):

    nlp = spacy.load("pt_core_news_sm")

    matcher = Matcher(nlp.vocab)

    pattern1 = [{'POS': 'NUM'}, {'POS': 'NOUN'}]
    pattern2 = [{'POS': 'DET'}, {'POS': 'NOUN'}]

    matcher.add("order", [pattern1, pattern2])

    doc = nlp(text)
    print([(t.text, t.pos_) for t in doc])

    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    print(spacy.util.filter_spans(spans))

    # for token in doc:
    #     print(f'{token.text:{8}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')