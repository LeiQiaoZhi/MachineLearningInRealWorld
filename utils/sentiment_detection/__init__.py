__all__ = ['read_tokens', 'load_reviews', 'split_data', 'clean_plot', 'print_binary_confusion_matrix']

from contextlib import contextmanager
from .tokenizer import read_tokens
from .data_loader import load_reviews, split_data, read_student_review_predictions
from .plots import best_fit, clean_plot, chart_plot
from .printer import print_binary_confusion_matrix, print_agreement_table


@contextmanager
def should_work(msg=True):
    try:
        yield
    except Exception as e:
        if not hasattr(e, 'should_work'):
            e.should_work = msg
        raise e


read_tokens = should_work(False)(read_tokens)
load_reviews = should_work(False)(load_reviews)
split_data = should_work(False)(split_data)
read_student_review_predictions = should_work(False)(read_student_review_predictions)
