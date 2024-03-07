import numpy as np
from collections import Counter

data = np.genfromtxt('./data/fma_small_genres.csv', delimiter=',', dtype=str)

genres = Counter(data[:, 1])

print(genres)