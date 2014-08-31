import numpy 
import scipy.stats
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

"""
We want to figure out the best way to estimate the w-th quantile of a set of numbers 
based on a sample. 
"""
numpy.set_printoptions(suppress=True, precision=3, linewidth=150)

def computeR(U, V,  w, indsPerRow=50): 
    """
    Given a matrix Z = UV.T compute a vector r such r[i] is the uth quantile 
    of the ith row of Z. We sample indsPerRow elements in each row and use that 
    for computing quantiles. Thus u=0 implies the smallest element and u=1 implies 
    the largest. 
    """
    m = U.shape[0]
    n = V.shape[0]
    r = numpy.zeros(m, numpy.float)
    tempRows = numpy.zeros((m, indsPerRow), numpy.float)
    colInds = numpy.zeros(indsPerRow, numpy.int)

    colInds = numpy.random.choice(n, indsPerRow, replace=True)
    tempRows = U.dot(V[colInds, :].T)
    r = numpy.percentile(tempRows, w*100.0, 1)
    
    return r  

def computeRScipy(U, V,  w, indsPerRow=50, alphap=0.4, betap=0.4): 
    """
    Given a matrix Z = UV.T compute a vector r such r[i] is the uth quantile 
    of the ith row of Z. We sample indsPerRow elements in each row and use that 
    for computing quantiles. Thus u=0 implies the smallest element and u=1 implies 
    the largest. 
    """
    m = U.shape[0]
    n = V.shape[0]
    r = numpy.zeros(m, numpy.float)
    tempRows = numpy.zeros((m, indsPerRow), numpy.float)
    colInds = numpy.zeros(indsPerRow, numpy.int)

    colInds = numpy.random.choice(n, indsPerRow, replace=True)
    tempRows = U.dot(V[colInds, :].T)
    #r = numpy.percentile(tempRows, w*100.0, 1)
    r = scipy.stats.mstats.mquantiles(tempRows, w, axis=1).ravel()
    
    return r  

m = 50 
n = 200 

u = 0.9 
w = 1-u 
numAucSamples = numpy.arange(5, 201, 5) 
k = 8
numRuns = 50

pvals = [(0, 1), (0.5, 0.5), (0, 0), (1, 1), (1.0/3, 1.0/3), (3.0/8, 3.0/8), (0.4, 0.4), (0.35, 0.35)]


errors = numpy.zeros((len(pvals)+1, numAucSamples.shape[0], numRuns))

for j in range(numRuns): 
    U = numpy.random.randn(m, k)
    V = numpy.random.randn(n, k)
    
    Z = U.dot(V.T)
    rReal = numpy.percentile(Z, w*100.0, 1)
    
    for i, aucSamples in enumerate(numAucSamples): 
        
        for ell, pval in enumerate(pvals): 
            alphap, betap = pval 
            r = computeRScipy(U, V, w, aucSamples, alphap, betap)
            errors[ell, i, j] = numpy.linalg.norm(rReal - r)
            
        r = computeR(U, V, w, aucSamples)
        errors[ell+1, i, j] = numpy.linalg.norm(rReal - r)

meanErrors = numpy.mean(errors, 2)
print(meanErrors)

for ell, pval in enumerate(pvals):
    plt.plot(numAucSamples, meanErrors[ell, :], label=str(pval))
plt.plot(numAucSamples, meanErrors[ell+1, :], label="original")


plt.legend()
plt.show()

#Best one is 0.35, 0.35 but not by much 
#u=0.5 u is a bit more stable 
#u=0.9 is less stable

#Pick the mean? 