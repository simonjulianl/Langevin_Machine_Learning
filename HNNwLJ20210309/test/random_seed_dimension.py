import torch

import numpy as np
import random
np.random.seed(0)
random.seed(43893324)

for i in range(4):

    pos = np.random.uniform(-0.5, 0.5, (2, 2))
    print(pos)
    # pos = np.expand_dims(pos, axis=0)
    # print(pos)