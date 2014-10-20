import numpy 
import logging
import sppy 
import sppy.io
import array 
import scipy.io
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer 
from sandbox.util.Sampling import Sampling 

class DatasetUtils(object): 
    """
    Some functions to load/generate datasets 
    """
    
    @staticmethod
    def syntheticDataset1(m=500, n=200, k=8, u=0.1, sd=0, noise=5): 
        """
        Create a simple synthetic dataset 
        """
        w = 1-u
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, sd=sd, csarray=True, verbose=True, indsPerRow=200)
        X = X + sppy.rand((m, n), noise/float(n), storagetype="row")
        X[X.nonzero()] = 1
        X.prune()
        X = SparseUtils.pruneMatrixRows(X, minNnzRows=10)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        U = U*s
        
        return X, U, V
    
    @staticmethod
    def syntheticDataset2(): 
        """
        Create a simple synthetic dataset using a power law distribution on users and items 
        """
        resultsDir = PathDefaults.getDataDir() + "syntheticRanking/"
        matrixFileName = resultsDir + "dataset1.mtx" 
        
        X = sppy.io.mmread(matrixFileName, storagetype="row")
        
        return X   
    
    @staticmethod
    def movieLens(minNnzRows=10, minNnzCols=2, quantile=90): 
        matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
        data = numpy.loadtxt(matrixFileName)
        X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row", dtype=numpy.int)
        X.put(numpy.array(data[:, 2]>3, numpy.int), numpy.array(data[:, 0]-1, numpy.int32), numpy.array(data[:, 1]-1, numpy.int32), init=True)
        #X = SparseUtilsCython.centerRowsCsarray(X)   
        #X[X.nonzero()] = X.values()>0
        X.prune()
        #maxNnz = numpy.percentile(X.sum(0), quantile)
        #X = SparseUtils.pruneMatrixCols(X, minNnz=minNnzCols, maxNnz=maxNnz)
        X = SparseUtils.pruneMatrixRowAndCols(X, minNnzRows, minNnzCols)
        logging.debug("Read file: " + matrixFileName)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))

        return X
        
    @staticmethod 
    def flixster(minNnzRows=10, minNnzCols=2, quantile=90): 
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
        
        X = SparseUtils.pruneMatrixRowAndCols(X, minNnzRows, minNnzCols)
        
        logging.debug("Read file: " + matrixFileName)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        
        #X = Sampling.sampleUsers(X, 1000)
        
        return X 

    @staticmethod         
    def mendeley(minNnzRows=10, minNnzCols=2, quantile=90, dataset="Doc", sigma=0.05, indicator=True):
        authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthor"+ dataset + "Matrix_sigma=" + str(sigma) + ".mtx"
        logging.debug("Reading file: " + authorAuthorFileName)
        X = sppy.io.mmread(authorAuthorFileName, storagetype="row")
        
        if indicator: 
            X[X.nonzero()] = 1
            X.prune()
        
        logging.debug("Raw non-zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        
        X = SparseUtils.pruneMatrixRowAndCols(X, minNnzRows, minNnzCols)
        
        logging.debug("Read file: " + authorAuthorFileName)
        logging.debug("Non-zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        
        return X
        
    @staticmethod         
    def mendeley2(minNnzRows=10, minNnzCols=2, quantile=90, dataset="Document"):
        authorAuthorFileName = PathDefaults.getDataDir() + "reference/author" + dataset + "Matrix.mtx"
        logging.debug("Reading file: " + authorAuthorFileName)
        X = sppy.io.mmread(authorAuthorFileName, storagetype="row")
                
        logging.debug("Raw non-zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        
        X = SparseUtils.pruneMatrixRowAndCols(X, minNnzRows, minNnzCols)
        
        logging.debug("Read file: " + authorAuthorFileName)
        logging.debug("Non-zero elements: " + str(X.nnz) + " shape: " + str(X.shape))
        
        return X 
        
        
    @staticmethod 
    def bookCrossing(minNnzRows=10, minNnzCols=2, quantile=90): 
        matrixFileName = PathDefaults.getDataDir() + "book-crossing/BX-Book-Ratings.csv" 
        matrixFile = open(matrixFileName)
        matrixFile.readline()
        userIndexer = IdIndexer("i")
        itemIndexer = IdIndexer("i")
        
        ratings = array.array("f")
        logging.debug("Loading ratings from " + matrixFileName)
        
        for i, line in enumerate(matrixFile):
            if i % 1000000 == 0: 
                logging.debug("Iteration: " + str(i))
            vals = line.split(";")
            
            field1 = vals[0].strip("\"")
            field2 = vals[1].strip("\"")
            field3 = int(vals[2].strip("\"\n\r"))            
            
            userIndexer.append(field1)
            itemIndexer.append(field2)
            ratings.append(field3)
                    
        rowInds = userIndexer.getArray()
        colInds = itemIndexer.getArray()
        ratings = numpy.array(ratings)
                
        X = sppy.csarray((len(userIndexer.getIdDict()), len(itemIndexer.getIdDict())), storagetype="row", dtype=numpy.int)
        X.put(numpy.array(numpy.logical_or(ratings>4, ratings==0), numpy.int), numpy.array(rowInds, numpy.int32), numpy.array(colInds, numpy.int32), init=True)
        X.prune()
        
        X = SparseUtils.pruneMatrixRowAndCols(X, minNnzRows, minNnzCols)
        
        logging.debug("Read file: " + matrixFileName)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))

        return X 
        
    @staticmethod 
    def epinions(minNnzRows=10, minNnzCols=2, quantile=90): 
        matrixFileName = PathDefaults.getDataDir() + "epinions/rating.mat" 
        A = scipy.io.loadmat(matrixFileName)["rating"]
        
        userIndexer = IdIndexer("i")
        itemIndexer = IdIndexer("i")        
        
        for i in range(A.shape[0]): 
            userIndexer.append(A[i, 0])
            itemIndexer.append(A[i, 1])


        rowInds = userIndexer.getArray()
        colInds = itemIndexer.getArray()
        ratings = A[:, 3]        
        
        X = sppy.csarray((len(userIndexer.getIdDict()), len(itemIndexer.getIdDict())), storagetype="row", dtype=numpy.int)
        X.put(numpy.array(ratings>3, numpy.int), numpy.array(rowInds, numpy.int32), numpy.array(colInds, numpy.int32), init=True)
        X.prune()
        
        X = SparseUtils.pruneMatrixRowAndCols(X, minNnzRows, minNnzCols)
        
        logging.debug("Read file: " + matrixFileName)
        logging.debug("Non zero elements: " + str(X.nnz) + " shape: " + str(X.shape))

        return X 