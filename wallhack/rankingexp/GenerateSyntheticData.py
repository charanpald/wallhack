import os
import sys 
import sppy.io
import numpy 
import logging
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)
m = 600 
n = 300 
k = 8
density = 0.1

X, U, V = SparseUtilsCython.generateSparseBinaryMatrixPL((m,n), k, density=density, alpha=1, csarray=True)
X = SparseUtils.pruneMatrixRows(X, minNnzRows=10)

resultsDir = PathDefaults.getDataDir() + "syntheticRanking/"

if not os.path.exists(resultsDir): 
    os.mkdir(resultsDir)

matrixFileName = resultsDir + "dataset1.mtx" 

sppy.io.mmwrite(matrixFileName, X)
logging.debug("Non-zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
logging.debug("Saved file: " + matrixFileName)

