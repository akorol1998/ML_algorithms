from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random as rd


def euclidean_distance(x:list, y:list):
	style.use("fivethirtyeight")
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
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	
	return vote_result, confidence


def our_test():
	df = pd.read_csv("actual_data.txt")
	df.replace('?', -99999, inplace=True)
	df.drop(['id'], 1, inplace=True)
	full_data = df.astype(float).values.tolist()
	rd.shuffle(full_data)


	test_size = 0.2
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	train_data = full_data[:-int(len(full_data) * test_size)]
	test_data = full_data[-int(len(full_data) * test_size):]


	correct = 0;
	total = 0
	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	for group in test_set:
		for data in test_set[group]:
			vote, confidence = k_nearest_neaighbors(train_set, data, k=5)
			if vote == group:
				correct += 1
			else:
				print("fails %s" % confidence)
			total += 1

	return "Accuracy is....{}".format(correct/total, confidence)


if __name__ == "__main__":
	print(our_test())
	