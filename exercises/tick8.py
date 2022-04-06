from utils.markov_models import load_dice_data
import os
from exercises.tick7 import estimate_hmm
import random, math,sys
# rmya2
from typing import List, Dict, Tuple

states = ['B','Z','F','W']

sys.setrecursionlimit(int(1e4))

def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    observed_sequence = ['B'] + observed_sequence + ['Z']
    table = [{} for _ in range(len(observed_sequence))] # memoization table for delta
    psi_table = [{} for _ in range(len(observed_sequence))]

    for si in states:
        delta(si,len(observed_sequence)-1,transition_probs,emission_probs,observed_sequence,table,psi_table)

    # backtrace
    ans = [None for _ in range(len(observed_sequence)-2)]
    s = 'Z' # current state in backtrace
    for t in range(len(ans)):
        s = psi_table[len(observed_sequence)-1-t][s]
        ans[-1-t] = s
    return ans

def delta(sj, time, transition_probs, emission_probs, observed_sequence,table,psi_table):
    if time == 0: # the starting state
        table[time][sj] = log(emission_probs[(sj,observed_sequence[time])])
    # checks if in memoization table
    if sj in table[time]: return table[time][sj]

    m = -math.inf # max delta
    mstate = '' # max state
    for si in states:
        previous_delta = delta(si,time-1,transition_probs,emission_probs,observed_sequence,table,psi_table)+\
            log(transition_probs[(si,sj)])+log(emission_probs[(sj,observed_sequence[time])])
        if previous_delta > m:
            m = previous_delta
            mstate = si
    # store result in memoization table
    table[time][sj] = m
    psi_table[time][sj] = mstate
    return m

# helper log function that deals with log(0)
def log(n):
    if n == 0: return -math.inf
    return math.log(n)

def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    # precision of 1 = times system correctly predicts 1 / times system predicts 1
    correct = system1 = 0
    for (p,t) in zip(pred,true):
        if not len(p) == len(t): print(f"pred and true length not equal {len(p)}, {len(t)}")
        for i in range(len(p)):
            if p[i] == t[i] and p[i] == 1:
                correct+=1
            if p[i] == 1:
                system1 += 1
    return correct/system1 if system1 > 0 else 0

def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    # recall of 1 = times system correctly predicts 1 / times truth is 1
    correct = true1 = 0
    for (p,t) in zip(pred,true):
        if not len(p) == len(t): print(f"pred and true length not equal {len(p)}, {len(t)}")
        for i in range(len(p)):
            if p[i] == t[i] and p[i] == 1:
                correct+=1
            if t[i] == 1:
                true1 += 1
    return correct/true1 if true1 > 0 else 0


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    precision = precision_score(pred,true)
    recall = recall_score(pred,true)
    return 2*precision*recall/(precision + recall) if precision + recall > 0 else 0


def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    # divide data into 10 folds
    d = {}
    n = 10
    random.shuffle(data)
    folds = []
    fold_size = len(data)//n
    for i in range(n):
        folds.append(data[i*fold_size:i*fold_size+fold_size])

    # each fold is a list of sequences
    # we use each fold as test data once
    for fold in folds:
        # get trainig data
        train = []
        for tf in folds:
            if not fold == tf:
                train += tf
        # train to get transition and emission matrices
        transition_probs, emission_probs = estimate_hmm(train)

        # get all predictions and truth for the test fold
        fold_pred = []
        fold_true = []
        for seq in fold:
            # get prediction for one observed sequence
            pred = viterbi(seq['observed'], transition_probs, emission_probs)
            fold_pred.append(pred)
            fold_true.append(seq['hidden'])

        bin_fold = [[1 if x=='W' else 0 for x in p] for p in fold_pred]
        bin_true = [[1 if x=='W' else 0 for x in t] for t in fold_true]

        precision = precision_score(bin_fold,bin_true)
        recall = recall_score(bin_fold,bin_true)
        f1 = f1_score(bin_fold,bin_true)

        d['precision'] = d.get('precision',0) + precision/n
        d['recall'] = d.get('recall',0) + recall/n
        d['f1'] = d.get('f1',0) + f1/n
    return d

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)


    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")



if __name__ == '__main__':
    main()
