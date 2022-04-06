from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, read_lexicon, predict_sentiment
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from exercises.tick4 import sign_test
import random
import numpy


def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    res = []
    fold_size = len(training_data)//n
    for i in range(n):
        res.append(training_data[i*fold_size:i*fold_size+fold_size])
    return res


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    pos = []
    neg = [] # pos and neg reviews in the entire training data
    for review in training_data:
        if review['sentiment'] > 0:
            pos.append(review)
        if review['sentiment'] < 0:
            neg.append(review)
    res = []
    fold_size = len(training_data)//n
    fold_pos = round(fold_size * len(pos)/len(training_data)) # number of pos in a fold
    fold_neg = fold_size - fold_pos
    print(f"len training data: {len(training_data)}, len pos: {len(pos)}")
    print(f"fold size: {fold_size}, fold pos: {fold_pos}, fold neg: {fold_neg}")
    for i in range(n):
        fold = pos[i*fold_pos:i*fold_pos+fold_pos] + neg[i*fold_neg:i*fold_neg+fold_neg]
        res.append(fold)
    return res

def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    acc = []
    for i in range(len(split_training_data)):
        test = split_training_data[i]
        train = [] # training data for each fold
        for j in range(len(split_training_data)):
            if j != i:
                train += split_training_data[j]
        # training
        class_priors = calculate_class_log_probabilities(train)
        smoothed_log_probabilities = calculate_smoothed_log_probabilities(train)
        preds_smoothed = [] # predicted results
        for review in [d['text'] for d in test]:
            pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
            preds_smoothed.append(pred)
        acc_smoothed = accuracy(preds_smoothed, [d['sentiment'] for d in test])
        acc.append(acc_smoothed)
    return acc

def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return sum(accuracies)/len(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    mean_accuracy = cross_validation_accuracy(accuracies)
    return sum([(acc-mean_accuracy)**2 for acc in accuracies])/len(accuracies)


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    true_pos =true_neg=false_pos=false_neg = 0
    for pred, actual in zip(predicted_sentiments,actual_sentiments):
        if pred == -1 and actual == -1:
            true_neg+=1
        elif pred==-1 and actual == 1:
            false_neg+=1
        elif pred==1 and actual==-1:
            false_pos+=1
        elif pred==1 and actual==1:
            true_pos+=1
    return [[true_pos,false_pos],[false_neg,true_neg]]

# def print_binary_confusion_matrix(confusion_matrix:List[List[int]]):
#     left_margin = 10
#     print(" "*left_margin+"actual:")
#     print(" "*left_margin+"pos  neg")
#     print("predicted: " + f"pos {confusion_matrix[0][0]} {confusion_matrix[0][1]}")
#     print(" "*left_margin + f"neg {confusion_matrix[1][0]} {confusion_matrix[1][1]}")

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    # Simple CLASSIFIER
    print("\nSIMPLE CLASSIFIER:")
    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')

    ### SOMETHING WRONG
    preds_heldout = [predict_sentiment(t, lexicon) for t in test_tokens]
    acc = accuracy(preds_heldout, test_sentiments)
    print(f"accuracy on held out data: {acc}")
    print_binary_confusion_matrix(confusion_matrix(preds_heldout, test_sentiments))

    simp_preds_recent = [predict_sentiment(t, lexicon) for t in recent_tokens]
    acc1 = accuracy(simp_preds_recent, recent_sentiments)
    print(f"accuracy on recent data: {acc1}")
    print_binary_confusion_matrix(confusion_matrix(simp_preds_recent, recent_sentiments))

    # Significance Test
    print("\nSignificance Test:")
    p = sign_test(recent_sentiments,simp_preds_recent,preds_recent)
    print(f"The p-value of the two-sided sign test for NBC and SimpleC on 2016 data is {p}")


if __name__ == '__main__':
    main()
