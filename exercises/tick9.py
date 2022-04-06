from utils.markov_models import load_bio_data, print_matrices
import os,sys
import random, math
from exercises.tick8 import recall_score, precision_score, f1_score

from typing import List, Dict, Tuple

# zy317

sys.setrecursionlimit(int(1e4))

hidden_states = ['B','Z','i','o','M']
observations = ['B,','Z','P', 'A', 'T', 'K', 'S', 'N', 'L', 'Q', 'M', 'G', 'Z', 'B', 'D', 'H', 'I', 'C', 'W', 'E', 'R', 'V', 'Y', 'F']
def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    # aij = number of all transitions from si to sj / number of all transitions from si to any state
    d = {}
    for si in hidden_states:
        for sj in hidden_states:
            if (si == 'B' and sj == 'Z') or si == 'Z' or sj == 'B':
                d[(si,sj)] = 0
            else:
                count = 0
                count_all = 0
                for seq in hidden_sequences:
                    previous = None
                    for s in seq:
                        if s == si:
                            count_all+=1
                        if s == sj and previous == si:
                            count+=1
                        previous = s
                d[(si,sj)] = count/count_all
    return d


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    # bi(kj) = number of emissions of oberservation oj in state si / number of any emission in state si
    d = {}
    for si in hidden_states:
        for oj in observations:
            count = count_all = 0
            for (s_seq,o_seq) in zip(hidden_sequences,observed_sequences):
                for (s,o) in zip(s_seq,o_seq):
                    if s == si:
                        count_all+=1
                    if s == si and o == oj:
                        count+=1
            d[(si,oj)] = count/count_all
    return d

def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)
    #print_matrices(transition_probs)
    #print_matrices(emission_probs)
    return [transition_probs, emission_probs]


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    observed_sequence = ['B'] + observed_sequence + ['Z']
    table = [{} for _ in range(len(observed_sequence))] # memoization table for delta
    psi_table = [{} for _ in range(len(observed_sequence))]

    for si in hidden_states:
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
    for si in hidden_states:
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


def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
    unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """
    scores = []

    train = training_data
    pseudo_data = []

    for t in range(num_iterations+1): # num_iterations + 1 cuz tester wants iteration 0 -- before any self training
        transition_probs, emission_probs = estimate_hmm_bio(training_data+pseudo_data)

        # prediction on unlabelled data
        predictions = []
        for sample in unlabeled_data:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            predictions.append(prediction)

        # updata pesudo_data
        pseudo_data = [{'observed':ob, 'hidden':pred} for (ob,pred) in zip(unlabeled_data,predictions)]

        # calculate score
        # same as those in main
        dev_predictions = []
        dev_observed_sequences = [x['observed'] for x in dev_data]
        dev_hidden_sequences = [x['hidden'] for x in dev_data]
        for sample in dev_observed_sequences:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            dev_predictions.append(prediction)
        predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in dev_predictions]
        dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

        p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

        print(f"Your precision for iteration {t} using the HMM: {p}")
        print(f"Your recall for iteration {t} using the HMM: {r}")
        print(f"Your F1 for iteration {t} using the HMM: {f1}\n")

        scores.append({'recall':r,"precision":p,"f1":f1})
    return scores


def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    clean_plot()
    r = [(i,score_list[i]['recall']) for i,iter in enumerate(score_list)]
    p = [(i,score_list[i]['precision']) for i,iter in enumerate(score_list)]
    f1 = [(i,score_list[i]['f1']) for i,iter in enumerate(score_list)]
    chart_plot(r,"score vs iteration","iteration","score",'recall')
    chart_plot(p,"score vs iteration","iteration","score",'precision')
    chart_plot(f1,"score vs iteration","iteration","score",'f1')



def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)

    visualize_scores(scores_each_iteration)



if __name__ == '__main__':
    main()
