"""
Create a small toy dataset which we will use for recommendation. 
"""

import logging 
import sys 
import numpy 
import numpy.linalg 
import scipy.io 
from sandbox.util.PathDefaults import PathDefaults 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

numExamples = 500
rank = 50 
 
A = numpy.random.rand(numExamples, numExamples)
A = A.dot(A.T)
s, U = numpy.linalg.eig(A)
U = U[:, 0:rank] 
#Make sure result is non-negative by taking the absolute value of single vectors 
U = numpy.abs(U)

B = numpy.random.rand(numExamples, numExamples)
B = B.dot(B.T)
s, V = numpy.linalg.eig(B)
V = V[:, 0:rank]
V = numpy.abs(V)

s = numpy.random.rand(rank)

X = (U*s).dot(V.T)

#Save matrix 
outputDir = PathDefaults.getOutputDir() + "erasm/"
fileName = outputDir + "Toy" 
scipy.io.mmwrite(fileName, X)

logging.debug("Saved to file " + fileName + ".mtx")

