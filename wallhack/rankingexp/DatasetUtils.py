import numpy 
import logging
import sppy 
import sppy.io
import array 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer 
from sandbox.util.Sampling import Sampling 

class DatasetUtils(object): 
    """
    Some functions to load/generate datasets 
    """
    
    @staticmethod
    def syntheticDataset1(m=500, n=200, k=8, u=0.1, sd=0): 
        """
        Create a simple synthetic dataset 
        """
        w = 1-u
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, sd=sd, csarray=True, verbose=True, indsPerRow=200)
        X.prune()
        X = SparseUtils.pruneMatrixRows(X, minNnzRows=10)
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

        return X
        
    @staticmethod 
    def flixster(): 
        matrixFileName = PathDefaults.getDataDir() + "flixster/Ratings.timed.txt" 
        matrixFile = open(matrixFileName)
        matrixFile.readline()
        userIndexer = IdIndexer("i")
        movieIndexer = IdIndexer("i")
        
        ratings = array.array("f")
        logging.debug("Loading ratings from " + matrixFileName)
        
        for i, line in enumerate(matrixFile):
            if i % 1000000 == 0: 
                logging.debug("Iteration: " + str(i))
            vals = line.split()
            
            userIndexer.append(vals[0])
            movieIndexer.append(vals[1])
            ratings.append(float(vals[2]))
        
        rowInds = userIndexer.getArray()
        colInds = movieIndexer.getArray()
        ratings = numpy.array(ratings)
        
        X = sppy.csarray((len(userIndexer.getIdDict()), len(movieIndexer.getIdDict())), storagetype="row", dtype=numpy.int)
        X.put(numpy.array(ratings>3, numpy.int), numpy.array(rowInds, numpy.int32), numpy.array(colInds, numpy.int32), init=True)
        X.prune()
        X = SparseUtils.pruneMatrixRows(X, minNnzRows=10)
        logging.debug("Read file: " + matrixFileName)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        
        X = Sampling.sampleUsers(X, 1000)
        
        return X 

    @staticmethod         
    def mendeley():
        authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
        logging.debug("Reading file: " + authorAuthorFileName)
        X = sppy.io.mmread(authorAuthorFileName, storagetype="row")
        (m, n) = X.shape
        logging.debug("Read file: " + authorAuthorFileName)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        
        return X 