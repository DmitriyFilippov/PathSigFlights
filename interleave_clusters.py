import random


def interleave(c1, c2):

    c1_labeled = list(map(lambda x: [x, 0], c1))
    c2_labeled = list(map(lambda x: [x, 1], c2))
    c_labeled = c1_labeled + c2_labeled
    random.shuffle(c_labeled)
    c = [i[0] for i in c_labeled]
    labels = [i[1] for i in c_labeled]
    return c, labels