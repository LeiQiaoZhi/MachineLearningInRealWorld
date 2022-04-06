import os
import glob
from typing import List, Dict, Union
import random


def load_reviews(path: str, include_nuance=False):
    review_sentiment = {}
    with open(os.path.join(path, 'review_sentiment'), encoding='utf-8') as f2:
        for line in f2.readlines():
            id, sentiment = line.strip().split('\t')
            if sentiment == 'POS':
                review_sentiment[id] = 1
            elif sentiment == 'NEG':
                review_sentiment[id] = -1
            elif sentiment == 'NEU' and include_nuance:
                review_sentiment[id] = 0

    reviews = glob.glob(os.path.join(path, 'reviews', '*'))
    reviews.sort()
    review_data = [{'filename': x, 'sentiment': review_sentiment[x.split(os.sep)[-1]]} for x in reviews
                   if x.split(os.sep)[-1] in review_sentiment.keys()]
    return review_data


def split_data(review_data: List[Dict[str, Union[List[str], int]]], seed=0):
    rand = random.Random(seed)
    training_set = []
    validation_set = []

    sentiment_set = sorted(list(set([x['sentiment'] for x in review_data])))
    number_of_elements = len(review_data)

    number_of_classes = len(sentiment_set)
    overall_number_with_test = number_of_elements/0.9
    VALIDATION_FRACTION = 0.1
    validation_size_per_label = round((VALIDATION_FRACTION * overall_number_with_test) / number_of_classes)

    label_paths = {}
    # Select all data points with the label and randomize order
    for label in sentiment_set:
        paths = [x['filename'] for x in review_data if x['sentiment'] == label]
        rand.shuffle(paths)
        label_paths[label] = paths

    # Assume balanced data. Get balanced 8:1:1 train:validation:test split
    for label in label_paths:
        for x in label_paths[label][:validation_size_per_label]:
            validation_set.append({'filename': x, 'sentiment': label})
        for x in label_paths[label][validation_size_per_label:]:
            training_set.append({'filename': x, 'sentiment': label})

    return [training_set, validation_set]


def read_student_review_predictions(path: str) -> List[Dict[int, int]]:
    """
    Reads in the student review csv file with the manual review sentiment predictions from task 1.

    @param path: path to the file
    @return: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being
    the review id and the value the predicted sentiment
    """
    agreement_table = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            content = line.strip().split(',')
            entry = {}
            for i, x in enumerate(content):
                entry[i] = 1 if x == 'Positive' else -1
            agreement_table.append(entry)
    return agreement_table
