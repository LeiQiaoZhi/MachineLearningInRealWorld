import os,math
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy

# hem52
def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    d = {1:0.0,-1:0.0,0:0.0}
    len_total = len(training_data)
    len_pos = len_neg = len_neutral = 0

    for review in training_data:
        if review['sentiment'] == 1:
            len_pos += 1
        elif review['sentiment'] == -1:
            len_neg += 1
        else:
            len_neutral += 1

    P_pos = len_pos / len_total
    P_neg = len_neg / len_total
    P_neutral = len_neutral / len_total

    d[1] = math.log(P_pos) if P_pos > 0 else -1e10
    d[-1] = math.log(P_neg) if P_pos > 0 else -1e10
    d[0] = math.log(P_neutral) if P_pos > 0 else -1e10
    #print(f"{P_pos},{P_neg},{P_neutral}")
    return d

def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    # dictionary {WORD:COUNT}
    dc_pos = {} # {'word':count of word in positive reviews}
    dc_neg = {}
    dc_neutral = {}

    # dictionary {WORD:PROBABILITY}
    d_pos = {} # {'word':P(w|POS)}
    d_neg = {}
    d_neutral = {}

    # initialise the dictionaries
    for review in training_data:
        for w in review['text']:
            if review['sentiment']==1:
                if w not in dc_pos:
                    dc_pos[w] = 1
                else:
                    dc_pos[w] = dc_pos[w] + 1
            elif review['sentiment']==-1:
                if w not in dc_neg:
                    dc_neg[w] = 1
                else:
                    dc_neg[w] = dc_neg[w]+1
            else:
                if w not in dc_neutral:
                    dc_neutral[w] = 1
                else:
                    dc_neutral[w] += 1

    all_words = set(list(dc_pos.keys())+list(dc_neg.keys())+list(dc_neutral.keys()))
    count_all = len(all_words) # count of all words in ALL reviews
    count_pos = sum(dc_pos.values())+count_all # count of appearance of all words in POS reviews
    count_neg = sum(dc_neg.values())+count_all # count of all words in NEG reviews
    count_neutral = sum(dc_neutral.values())+count_all
    #print(f"{count_all},{count_pos},{count_neg},{count_neutral}")

    # calculate the probabilities
    for w in all_words:
        if w not in d_pos:
            pos = dc_pos.get(w,0)+1
            d_pos[w] = math.log((pos) / count_pos)
        if w not in d_neg:
            neg = dc_neg.get(w,0)+1
            d_neg[w] = math.log((neg) / count_neg)
        if w not in d_neutral:
            neutral = dc_neutral.get(w,0)+1
            d_neutral[w] = math.log((neutral) / count_neutral)

    d = {1:d_pos,-1:d_neg,0:d_neutral}
    #print(f"good: count:{dc_pos['good'],dc_neg['good']} prob:{d_pos['good']}")
    return d


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
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
    print(f"correct:{correct},total:{len(pred)}, true:{len(true)}")
    return correct/min(len(pred),len(true))


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    p_pos = class_log_probabilities[1] + sum([log_probabilities[1].get(w,0) for w in review])
    p_neg = class_log_probabilities[-1] + sum([log_probabilities[-1].get(w,0) for w in review])
    p_neutral = class_log_probabilities[0] + sum([log_probabilities[0].get(w,0) for w in review])

    # find the class with highest probability
    m = max(p_pos,p_neg,p_neutral)
    if p_pos == m:
        return 1
    elif p_neg == m:
        return -1
    elif p_neutral == m:
        return 0
    else:
        raise Exception("Error with finding the class with the max probability")


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    pas = [] # list of all the Pa for all reviews
    pes = {} # {sentiment:total number in all reviews}
    total = 0 # total number of raters
    n = len(agreement_table) # number of reviews

    for review in agreement_table.values(): # review {sentiment:number of ratings}
        agree = 0 # number of pairs in agreement
        total = sum(review.values())
        for sentiment in review.keys():
            agree += comb(review[sentiment],2) if review[sentiment]>=2 else 0
            pes[sentiment] = pes.get(sentiment,0) + review[sentiment]
        pas.append(agree/comb(total,2))

    # take average
    pa = sum(pas)/len(pas)
    print(f"{pes}, total: {total}, n: {n}")
    pe = sum([(pes[s]/(total*n))**2 for s in pes.keys()])
    print(f"Pa: {pa} Pe: {pe}")
    kappa = (pa-pe)/(1-pe)
    return kappa

def comb(n,k):
    return (math.factorial(n))//(math.factorial(k)*math.factorial(n-k))

def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    d = {}
    for review_number in review_predictions[0]:
        d[review_number] = {1:0,-1:0}
    for student in review_predictions:
        for review_number in student.keys():
            student_prediction = student[review_number]
            if student_prediction in d[review_number]:
                d[review_number][student_prediction] += 1
    return d

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data[0:1000]]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2022.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
