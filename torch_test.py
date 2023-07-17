from collections import deque
import numpy as np
import torch
import torch.nn.functional as F


def sb(value, index):
    a = index % 5
    if a == 0 or a == 1:
        return True


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    b = np.array(a).flatten()
    c = [x for x in a]
    # c.reverse()
    print(a)
    print(c)
    a[::2] = c[1::2]
    a[1::2] = c[0::2]
    print(a)
