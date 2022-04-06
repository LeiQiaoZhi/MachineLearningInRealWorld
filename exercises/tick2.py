from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
import math

def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior probability
        {1:logP(POS),-1:logP(NEG)}
    """
    d = {1:0.0,-1:0.0}
    len_total = len(training_data)
    len_pos = 0
    for review in training_data:
        if review['sentiment'] == 1:
            len_pos += 1
    len_neg = len_total - len_pos
    P_pos = len_pos / len_total
    P_neg = len_neg / len_total
    d[1] = math.log(P_pos)
    d[-1] = math.log(P_neg)
    return d


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    P(w|POS) = count of w with POS reviews / count of all words in POS reviews
    """
    d = {} # dictionary to return
    dc_pos = {} # {'word':count of word in positive reviews}
    dc_neg = {}
    d_pos = {} # {'word':P(w|POS)}
    d_neg = {}

    # count word occurances
    for review in training_data:
        for w in review['text']:
            if review['sentiment']==1:
                if w not in dc_pos:
                    dc_pos[w] = 1
                else:
                    dc_pos[w] = dc_pos[w] + 1
            else:
                if w not in dc_neg:
                    dc_neg[w] = 1
                else:
                    dc_neg[w] = dc_neg[w]+1

    count_pos = sum(dc_pos.values()) # count of appearance of all words in POS reviews
    count_neg = sum(dc_neg.values()) # count of all words in NEG reviews

    # calculate the probabilities
    # we ignore words that only appear in one class of review
    for w in dc_pos.keys():
        if w not in d_pos:
            d_pos[w] = math.log(dc_pos[w] / count_pos)
    for w in dc_neg.keys():
        if w not in d_neg:
            d_neg[w] = math.log(dc_neg[w] / count_neg)

    d = {1:d_pos,-1:d_neg}
    return d


def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    d = {}
    dc_pos = {} # {'word':count of word in positive reviews}
    dc_neg = {}
    d_pos = {} # {'word':P(w|POS)}
    d_neg = {}

    # initialise the dictionaries
    for review in training_data:
        for w in review['text']:
            if review['sentiment']==1:
                if w not in dc_pos:
                    dc_pos[w] = 1
                else:
                    dc_pos[w] = dc_pos[w] + 1
            else:
                if w not in dc_neg:
                    dc_neg[w] = 1
                else:
                    dc_neg[w] = dc_neg[w]+1

    all_words = set(list(dc_pos.keys())+list(dc_neg.keys()))
    count_all = len(all_words) # count of all words in ALL reviews
    count_pos = sum(dc_pos.values())+count_all # count of appearance of all words in POS reviews
    count_neg = sum(dc_neg.values())+count_all # count of all words in NEG reviews

    # calculate the probabilities
    for w in all_words:
        if w not in d_pos:
            pos = dc_pos.get(w,1)
            d_pos[w] = math.log((pos) / count_pos)
        if w not in d_neg:
            neg = dc_neg.get(w,1)
            d_neg[w] = math.log((neg) / count_neg)

    d = {1:d_pos,-1:d_neg}
    #print(f"good: count:{dc_pos['good'],dc_neg['good']} prob:{d_pos['good']}")
    return d


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    # probability of being positive
    p_pos = class_log_probabilities[1] + sum([log_probabilities[1].get(w,0) for w in review])
    p_neg = class_log_probabilities[-1] + sum([log_probabilities[-1].get(w,0) for w in review])
    s = 1 if p_pos > p_neg else -1
    return s


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    from exercises.tick1 import accuracy, predict_sentiment, read_lexicon

    # extracting data
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    dev_tokenized_data = [read_tokens(fn['filename']) for fn in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    # SIMPLE CLASSIFIER
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")


    # NB CLASSIFIER
    class_priors = calculate_class_log_probabilities(train_tokenized_data)

    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)

    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")

if __name__ == '__main__':
    main()
