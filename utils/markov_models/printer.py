from typing import Dict, Tuple


def print_matrices(probs: Dict[Tuple[str, str], float]) -> None:
    """
    Visualizes the probability matrices in the cosole in ascii.

    @param probs: A dictionary from a state tuple to a probability.
    @return: None
    """
    states_x1 = sorted(set([x[0] for x in probs]))
    states_x2 = sorted(set([x[1] for x in probs]))
    print('#' + ' '.join([x.center(5) for x in states_x2]))
    for x1 in states_x1:
        print(x1 + ' ' + ' '.join([f"{probs[(x1,x2)]:.3f}".center(5) for x2 in states_x2]))
