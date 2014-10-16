import numpy 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from math import exp, log 

"""
We just plot the loss functions. 
"""

n = 100
beta = 5
x = numpy.linspace(-1, 2, n)

y1 = numpy.zeros(n)
y2 = numpy.zeros(n)
y3 = numpy.zeros(n)
y4 = numpy.zeros(n)

for i in range(n): 
    y1[i] = max(0, 1-x[i])**2
    y2[i] = (1-x[i])**2
    y3[i] = 1/(1+exp(-beta*x[i]))
    y4[i] = log(1/(1+exp(-beta*x[i])))
    

plt.plot(x, y1, "k-", label="hinge")
plt.plot(x, y2, "k--", label="square")
plt.plot(x, y3, "k-.", label="sigmoid")
plt.plot(x, y4, "k:", label="logistic")
plt.xlabel("x")
plt.ylabel("L(x)")
plt.legend(loc="lower right")

plt.show()
    