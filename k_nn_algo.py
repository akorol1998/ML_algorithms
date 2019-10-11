#Euclidian Distance - юклідіан дістанс)
# It is  sum of (ai -pi)ˆ2 where i is one of the dimenssions


# Basicaly it is:
# sum = 0
# n = number of dimension
# for i in n:
# 	sum += (ai - pi)**2
# Euclidian distan
from math import sqrtce = sum**0.5
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use("fivethirtyeight")

def euclidean_distance(x:list, y:list):
	if len(x) != len(y):
	    raise Errors
	else:
	    return sqrt(sum([(x1 - y1) **2 for x1, y1 in zip(x, y)])) 

def k_nearest_neaighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn("K is set to a value less than total voting groups! Beach!")
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result	


if __name__ == "__main__":
	dataset = {'k': [[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
	new_features = [5,7]
	res = k_nearest_neaighbors(dataset, new_features)
	[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
	plt.scatter(new_features[0], new_features[1], s=150, color=res)
	plt.show()