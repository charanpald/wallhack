
import numpy 
import matplotlib.pyplot as plt 
from wallhack.rankingexp.DatasetUtils import DatasetUtils
"""
Do some basic analysis on the recommendation datasets. 
"""

X = DatasetUtils.movieLens()
print(X.shape)

itemCounts = X.sum(0)
print(itemCounts.shape)

p = numpy.percentile(itemCounts, 90)

popItems = itemCounts > p
unpopItems = itemCounts <= p

print((popItems).sum())
print((unpopItems).sum())

print(itemCounts[popItems].sum())
print(itemCounts[unpopItems].sum())

plt.hist(itemCounts[unpopItems])
plt.show()