import os
import glob
from typing import List, Dict, Union


def load_dice_data(path:str) -> List[Dict[str, List[str]]]:
    """
    Loads the dice dataset from the path

    @param path: path to the dice data folder
    @return: A list of dictionaries with a field ('observed') for the observed sequence and one for the hidden sequence ('hidden'), each encoded as a list of strings.
    Observations are encoded as a string: '1', '2', ..., '6'
    Hidden states are encoded as a string: 'F': FAIR, 'W': WEIGHTED;
    """
    dice_data = []
    dice_files = glob.glob(os.path.join(path, '*'))
    dice_files.sort()
    for file in dice_files:
        with open(file, encoding='utf-8') as f:
            content = f.readlines()
            observed = content[0].strip()
            hidden = content[1].strip()
            dice_data.append({'observed': list(observed), 'hidden': list(hidden)})
    return dice_data


def load_bio_data(path:str) -> List[Dict[str, List[str]]]:
    """
    Loads the biological dataset from the path

    @param path: path to the biodata file
    @return: A list of dictionaries with a field ('observed') for the observed sequence and one for the hidden sequence ('hidden'), each encoded as a list of strings.
    Observations are encoded as a string: 'R': ARG, 'H' : HIS, 'K': LYS, 'D': ASP, 'E': GLU, 'S': SER, 'T': THR, 'N': ASN, 'Q': GLN, 'C': CYS, 'U': SEC, G: GLY, 'P': PRO, 'A': ALA, 'V': VAL, 'I': ILE, 'L': LEU, 'M': MET, 'F': PHE, 'Y': TYR, 'W': TRP;
    Hidden states are encoded as a string: 'i': INSIDE, 'o': OUTSIDE, 'M': MEMBRANE
    """
    bio_data = []
    with open(os.path.join(path), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 3):
            bio_data.append({'observed': list(content[i].strip())[1:], 'hidden': list(content[i + 1].strip())})
    return bio_data
