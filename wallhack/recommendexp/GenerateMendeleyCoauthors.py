import array
import numpy
import scipy.sparse
import scipy.io
import logging
import sys 
from apgl.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer
from math import ceil 
import sppy 
import sppy.io
from apgl.util.ProfileUtils import ProfileUtils 
from math import sqrt
#import matplotlib
#matplotlib.use("GTK3Agg")
#import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

authorDocFileName = PathDefaults.getDataDir() + "reference/authorDocumentMatrix.mtx"

def writeAuthorDocMatrix(): 
    fileName = PathDefaults.getDataDir() + "reference/author_document_count"
    
    fileObj = open(fileName)
    
    authorIndex = IdIndexer()
    docIndex = IdIndexer()
    scores = array.array("i")
    
    for i, line in enumerate(fileObj):
        if i % 500000 == 0: 
            logging.debug(i)
        vals = line.split()
        #logging.debug(vals[0], vals[1], vals[2])
        
        authorIndex.append(vals[1])
        docIndex.append(vals[0])
        
        score = int(vals[2])
        scores.append(int(sqrt(score)))
    
    
    rowInds = numpy.array(authorIndex.getArray())
    colInds = numpy.array(docIndex.getArray())
    
    Y = scipy.sparse.csr_matrix((scores, (rowInds, colInds)))
        
    scipy.io.mmwrite(authorDocFileName, Y)
    logging.debug("Saved matrix to " + authorDocFileName)
    
def writeAuthorAuthorMatrix(): 
    
    Y = sppy.io.mmread(authorDocFileName, storagetype="row")
    logging.debug("Read file: " + authorDocFileName)
    logging.debug("Size of input :" + str(Y.shape))

    Y = sppy.csarray(Y, dtype=numpy.float, storagetype="row")
    #Y = Y[0:10000, :]
    
    invNorms = 1/numpy.sqrt((Y.power(2).sum(1)))
    Z = sppy.diag(invNorms, storagetype="row")
    Y = Z.dot(Y)
    
    sigma = 0.05
    blocksize = 500
    
    C = sppy.csarray((Y.shape[0], Y.shape[0]), storagetype="row")
    numBlocks = int(ceil(C.shape[0]/float(blocksize)))
    logging.debug("Number of blocks " + str(numBlocks))
    
    for i in range(numBlocks): 
        logging.debug("Iteration: " + str(i))
        
        endInd = min(Y.shape[0], (i+1)*blocksize)
                
        #tempY = Y.submatrix(i*blocksize, 0, endInd-i*blocksize, Y.shape[1])
        #tempC = tempY.dot(Y.T)
        tempC = Y[i*blocksize:endInd, :].dot(Y.T)
        tempC.clip(sigma, 1.0)

        rowInds, colInds = tempC.nonzero()
        rowInds += i*blocksize

        C.put(tempC.values(), rowInds, colInds)
    
    C[numpy.arange(Y.shape[0]), numpy.arange(Y.shape[0])] = 0 
    C.prune()    
    
    authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
    sppy.io.mmwrite(authorAuthorFileName, C)
    logging.debug("Saved matrix to " + authorAuthorFileName)
    logging.debug("Final size of C " + str(C.shape) + " with " + str(C.nnz) + " nonzeros")
    
#writeAuthorDocMatrix()
writeAuthorAuthorMatrix()   