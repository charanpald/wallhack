import logging 
import sys 
import numpy 
import matplotlib 
import powerlaw 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from wallhack.rankingexp.DatasetUtils import DatasetUtils
"""
Do some basic analysis on the recommendation datasets. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#X, U, V = DatasetUtils.syntheticDataset1()
#X = DatasetUtils.syntheticDataset2()
X = DatasetUtils.movieLens(quantile=100)
#X = DatasetUtils.flixster(quantile=100)
#X = DatasetUtils.mendeley(quantile=50)
print(X.shape)
m, n = X.shape

userCounts = X.sum(1)
itemCounts = X.sum(0)


results = powerlaw.Fit(itemCounts, discrete=True, xmax=n)
print(results.power_law.alpha)
print(results.power_law.xmin)
print(results.power_law.xmax)

u = 5 
p = numpy.percentile(itemCounts, 100-u)

popItems = itemCounts > p
unpopItems = itemCounts <= p

print(popItems.sum(), unpopItems.sum())
print(itemCounts[popItems].sum())
print(itemCounts[unpopItems].sum())

plt.figure(0)
plt.hist(itemCounts, bins=50, log=True)
plt.xlabel("users")
plt.ylabel("log frequency")

sortedCounts = numpy.flipud(numpy.sort(itemCounts))/numpy.sum(itemCounts)
sortedCounts = numpy.cumsum(sortedCounts)
plt.figure(1)
plt.plot(sortedCounts)
plt.xlabel("users")
plt.ylabel("cum probability")

plt.figure(2)
plt.hist(itemCounts[unpopItems])
plt.xlabel("users")
plt.ylabel("frequency")

plt.figure(3)
plt.hist(userCounts, bins=50, log=True)
plt.xlabel("items")
plt.ylabel("log frequency")
plt.show()
