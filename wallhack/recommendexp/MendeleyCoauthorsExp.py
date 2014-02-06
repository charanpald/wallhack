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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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
        scores.append(int(vals[2]))
    
    
    rowInds = numpy.array(authorIndex.getArray())
    colInds = numpy.array(docIndex.getArray())
    
    Y = scipy.sparse.csr_matrix((scores, (rowInds, colInds)))
    
    logging.debug(Y.shape, Y.nnz)
    
    outFileName = PathDefaults.getDataDir() + "reference/authorDocumentMatrix.mtx" 
    scipy.io.mmwrite(outFileName, Y)
    logging.debug("Saved matrix to " + outFileName)
    
def writeAuthorAuthorMatrix(): 
    matrixFileName = PathDefaults.getDataDir() + "reference/authorDocumentMatrix.mtx"
    Y = sppy.io.mmread(matrixFileName, storagetype="row")
    logging.debug("Read file: " + matrixFileName)
    logging.debug("Size of input :" + str(Y.shape))

    Y = sppy.csarray(Y, dtype=numpy.float, storagetype="row")
    #Y = Y[0:2000, :]
    
    invNorms = 1/numpy.sqrt((Y.power(2).sum(1)))
    Z = sppy.diag(invNorms, storagetype="row")
    Y = Z.dot(Y)
    
    sigma = 0.1
    blocksize = 500
    
    C = sppy.csarray((Y.shape[0], Y.shape[0]), storagetype="row")
    numBlocks = int(ceil(C.shape[0]/float(blocksize)))
    logging.debug("Num blocks " + str(numBlocks))
    
    for i in range(numBlocks): 
        logging.debug("Iteration: " + str(i))
        
        endInd = min(Y.shape[0], (i+1)*blocksize)
                
        tempC = Y[i*blocksize:endInd, :].dot(Y.T)
        tempC.clip(sigma, 1.0)

        rowInds, colInds = tempC.nonzero()
        rowInds += i*blocksize

        C.put(tempC.values(), rowInds, colInds, init=True)
    
    outFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
    sppy.io.mmwrite(outFileName, C)
    logging.debug("Saved matrix to " + outFileName)
    logging.debug("Final size of C " + str(C.shape) + " with " + str(C.nnz) + " nonzeros")
    
    """
    #Now save full matrix again without blocking 
    C = Y.dot(Y.T)
    C.clip(sigma, 1.0)
    
    outFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix2.mtx" 
    sppy.io.mmwrite(outFileName, C)
    logging.debug("Saved matrix to " + outFileName)
    """
    
#writeAuthorDocMatrix()
#writeAuthorAuthorMatrix()   
ProfileUtils.profile("writeAuthorAuthorMatrix()", globals(), locals())     
        