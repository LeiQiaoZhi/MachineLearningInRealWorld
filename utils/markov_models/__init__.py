__all__ = ['read_tokens', 'load_reviews', 'split_data', 'clean_plot']

from contextlib import contextmanager
from .data_loader import load_dice_data, load_bio_data
from .printer import print_matrices


@contextmanager
def should_work(msg=True):
    try:
        yield
    except Exception as e:
        if not hasattr(e, 'should_work'):
            e.should_work = msg
        raise e


load_dice_data = should_work(False)(load_dice_data)
load_bio_data = should_work(False)(load_bio_data)
print_matrices = should_work(False)(print_matrices)
