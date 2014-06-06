import numpy 
import logging
import sppy 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults 

class DatasetUtils(object): 
    """
    Some functions to load/generate datasets 
    """
    
    @staticmethod
    def syntheticDataset1(): 
        m = 500
        n = 200
        k = 8 
        u = 20.0/n
        w = 1-u
        X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        U = U*s
        
        return X, U, V
       
    @staticmethod
    def movieLens(): 
        matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
        data = numpy.loadtxt(matrixFileName)
        X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row", dtype=numpy.int)
        X.put(numpy.array(data[:, 2]>3, numpy.int), numpy.array(data[:, 0]-1, numpy.int32), numpy.array(data[:, 1]-1, numpy.int32), init=True)
        X.prune()
        X = SparseUtils.pruneMatrixRows(X, minNnzRows=10)
        logging.debug("Read file: " + matrixFileName)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        (m, n) = X.shape
        
        return X