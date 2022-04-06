import nltk


def read_tokens(filename):
    with open(filename, encoding='utf-8') as f:
        txt = f.readlines()
    words = []
    for line in txt:
        words.extend(nltk.tokenize.word_tokenize(line.strip()))

    return [w.lower() for w in words]
