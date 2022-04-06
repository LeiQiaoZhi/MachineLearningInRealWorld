import typing
import numpy as np
from utils.sentiment_detection import read_tokens, load_reviews


def read_lexicon(filename: str) -> typing.Dict[str, int]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    """
    d = {}
    f = open(filename,'r')
    for line in f:
        words = line.split()
        word = words[0].replace('word=','')
        intensity = words[1].replace('intensity=','')
        polarity = words[2].replace('polarity=','')
        sentiment = (1 if polarity=='positive' else -1)# * (4 if intensity=='strong' else 1)
        d[word] = sentiment
    return d



def predict_sentiment(tokens: typing.List[str], lexicon: typing.Dict[str, int]) -> int:
    """
    Given a list of tokens from a tokenized review and a lexicon, determine whether the sentiment of each review in the test set is
    positive or negative based on whether there are more positive or negative words.

    @param tokens: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    s = 0
    for token in tokens:
        if token in lexicon:
            sentiment = lexicon[token]
            s += sentiment
    return 1 if s >= 0 else -1


def accuracy(pred: typing.List[int], true: typing.List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    correct = 0
    for p, t in zip(pred,true):
        if p == t:
            correct+=1
    print(f"correct:{correct},total:{len(pred)}")
    return correct/len(pred)

def predict_better_sentiment(tokens,lexicon):
    s = 0
    for token in tokens:
        if token in lexicon:
            sentiment = lexicon[token]
            s += sentiment
    return s

def predict_sentiment_improved(tokens: typing.List[str], lexicon: typing.Dict[str, int]) -> int:
    """
    Use the training data to improve your classifier, perhaps by choosing an offset for the classifier cutoff which
    works better than 0.

    @param tokens: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1, -1 for positive and negative sentiments, respectively).
    """
    s = 0
    for token in tokens:
        if token in lexicon:
            sentiment = lexicon[token]
            s += sentiment
    return 1 if s >= 8.5 else -1


def main():
    """
    Check your code locally (from the root director 'mlrd') by calling:
    PYTHONPATH='.' python3.6 exercises/tick1.py
    """
    review_data = load_reviews('data/sentiment_detection/reviews')
    tokenized_data = [read_tokens(fn['filename']) for fn in review_data]

    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')

    pred1 = [predict_sentiment(t, lexicon) for t in tokenized_data]
    acc1 = accuracy(pred1, [x['sentiment'] for x in review_data])
    print(f"Your accuracy: {acc1}")

    # pred_better = np.array([predict_better_sentiment(t, lexicon) for t in tokenized_data])
    # pos = pred_better[np.array([x['sentiment'] for x in review_data])]
    # neg = pred_better[~np.array([x['sentiment'] for x in review_data])]
    # print(f"avg pos s: {pos.mean()}")
    # print(f"avg neg s: {neg.mean()}")

    pred2 = [predict_sentiment_improved(t, lexicon) for t in tokenized_data]
    acc2 = accuracy(pred2, [x['sentiment'] for x in review_data])
    print(f"Your improved accuracy: {acc2}")


if __name__ == '__main__':
    main()
