import numpy 
import scipy.stats
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from sandbox.util.SparseUtilsCython import SparseUtilsCython

"""
We want to figure out the best way to estimate the w-th quantile of a set of numbers 
based on a sample. 
"""
numpy.set_printoptions(suppress=True, precision=3, linewidth=150)

def computeR(U, V,  indsPerRow=50, func=numpy.mean, arg=None): 
    """
    Compute r as the mean. 
    """
    m = U.shape[0]
    n = V.shape[0]
    r = numpy.zeros(m, numpy.float)
    tempRows = numpy.zeros((m, indsPerRow), numpy.float)
    colInds = numpy.zeros(indsPerRow, numpy.int)

    colInds = numpy.random.choice(n, indsPerRow, replace=True)
    tempRows = U.dot(V[colInds, :].T)
    
    if arg == None: 
        r = func(tempRows)
    else: 
        r = func(tempRows, arg)
    
    return r  
    


m = 50 
n = 200 
numAucSamples = numpy.arange(5, 201, 5) 
k = 8
numRuns = 50
numMethods = 5
u = 0.1 
w = 1-u
errors = numpy.zeros((numMethods, numAucSamples.shape[0], numRuns))

for j in range(numRuns): 
    U = numpy.random.randn(m, k)
    V = numpy.random.randn(n, k)
    
    Z = U.dot(V.T)
    rReal = numpy.mean(Z, 1)
    
    for i, aucSamples in enumerate(numAucSamples): 
        r = computeR(U, V, aucSamples, numpy.mean)
        rReal = numpy.mean(Z, 1)
        errors[0, i, j] = numpy.linalg.norm(rReal - r)
        
        r = computeR(U, V, aucSamples, numpy.median)
        rReal = numpy.median(Z, 1)
        errors[1, i, j] = numpy.linalg.norm(rReal - r)
        
        r = computeR(U, V, aucSamples, numpy.min, 1)
        rReal = numpy.min(Z, 1)
        errors[2, i, j] = numpy.linalg.norm(rReal - r)

        r = computeR(U, V, aucSamples, numpy.max, 1)
        rReal = numpy.max(Z, 1)
        errors[3, i, j] = numpy.linalg.norm(rReal - r)        
        
        r = SparseUtilsCython.computeR(U, V, w, aucSamples)
        rReal = numpy.percentile(Z, w*100.0, 1)
        errors[4, i, j] = numpy.linalg.norm(rReal - r)
            
meanErrors = numpy.mean(errors, 2)
print(meanErrors)


plt.plot(numAucSamples, meanErrors[0, :], label="mean")
plt.plot(numAucSamples, meanErrors[1, :], label="median")
plt.plot(numAucSamples, meanErrors[2, :], label="min")
plt.plot(numAucSamples, meanErrors[3, :], label="max")
plt.plot(numAucSamples, meanErrors[4, :], label="u=0.1")

plt.legend()
plt.show()

#Mean is most stable 
