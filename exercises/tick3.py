from utils.sentiment_detection import clean_plot, read_tokens, chart_plot,best_fit
from typing import List, Tuple, Callable
import os
import glob
import math

def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    m,c = best_fit(token_frequencies_log,token_frequencies)
    print(f"best fit line: m:{m},c:{c}")
    print(f"alpha:{-m},k:{math.e**c}")
    return lambda x : m*x+c


# first task
def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    # get a list of all words from review files in datapath
    reviewfs = glob.glob(os.path.join(dataset_path, '*'))
    reviews = [read_tokens(f) for f in reviewfs]

    # process the words
    dc = {} # dictionary {word:count}
    for review in reviews:
        for word in review:
            if word not in dc:
                dc[word] = 1
            else:
                dc[word] += 1
    # sort
    l = [(k,v) for k,v in dc.items()] # list of tuples
    l = sorted(l,key=getKey,reverse=True)
    return l

def getKey(x):
    return x[1]

# second task
def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    # [(rank,frequency)]
    rank_frequency = [((r+1),f) for r,(w,f) in enumerate(frequencies)]
    chart_plot(rank_frequency,"word rank vs frequency","word rank","frequency")


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    selected_words = ['satisfying','classic','bland','lacking','well','fun','ironic','best','dry','annoying']
    selected_frequency = []
    for r,(w,f) in enumerate(frequencies):
        if w in selected_words:
            selected_frequency.append(((r+1),f))
    print(selected_frequency)
    chart_plot(selected_frequency,"word rank vs frequency","word rank","frequency")


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    # [(rank,frequency)]
    rank_frequency = [((r+1),f) for r,(w,f) in enumerate(frequencies)]
    rank_frequency_log = [(math.log(r),math.log(f)) for r,f in rank_frequency]
    chart_plot(rank_frequency_log,"zipf","word rank(log)","frequency(log)")

    # best fit
    #m,c = best_fit(rank_frequency_log,rank_frequency)
    f = estimate_zipf(rank_frequency_log,rank_frequency)
    x = rank_frequency_log[-1][0]
    chart_plot([(0,f(0)),(x,f(x))],"zipf","word rank(log)","frequency(log)")


def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    # get a list of all words from review files in datapath
    reviewfs = glob.glob(os.path.join(dataset_path, '*'))
    reviews = [read_tokens(f) for f in reviewfs]

    token_type = []
    types = set([]) # set of types
    wc = 0 # token count
    p = 0 # current power
    for review in reviews:
        for w in review:
            types.add(w)
            wc+=1
            if wc == pow(2,p):
                p+=1
                token_type.append((wc,len(types)))
    return token_type


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    data = [(math.log(token),math.log(type)) for (token,type) in type_counts]
    chart_plot(data,"heap","number of tokens(log)","number of types(log)")


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
