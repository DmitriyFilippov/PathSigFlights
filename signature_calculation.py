import sys
sys.path.append("..")
import esig
import numpy as np

import preprocessing

from hyperparameters import hyperparameters


esig.set_backend("iisignature")


def calculate_signatures(split_stream, hps = hyperparameters):
    depth = hps.depth
    signatures = []


    for stream in split_stream:
        stream = np.array(stream)
        stream = preprocessing.preprocess_stream(stream, hps)
        sig = esig.stream2sig(stream, depth)
        sig = preprocessing.preprocess_signature(sig, hps)
        signatures.append(sig)

    return signatures
