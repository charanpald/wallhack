import array
import numpy
import scipy.sparse
import scipy.io
import logging
import sys 
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer
from math import ceil 
import sppy 
import sppy.io
from sandbox.util.ProfileUtils import ProfileUtils 
from math import sqrt
import os.path

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

authorDocFileName = PathDefaults.getDataDir() + "reference/authorDocumentMatrix.mtx"

def writeAuthorDocMatrix(): 
    fileName = PathDefaults.getDataDir() + "reference/author_document_count"
    
    
    if not os.path.isfile(authorDocFileName): 
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
    else: 
        logging.debug("File exists: " + authorDocFileName)
    
def writeAuthorAuthorMatrix(): 
    authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
    
    if not os.path.isfile(authorAuthorFileName): 
        Y = sppy.io.mmread(authorDocFileName, storagetype="row")
        logging.debug("Read file: " + authorDocFileName)
        logging.debug("Size of input: " + str(Y.shape))
    
        Y = sppy.csarray(Y, dtype=numpy.float, storagetype="row")
        #Y = Y[0:1000, :]
        
        invNorms = 1/numpy.sqrt((Y.power(2).sum(1)))
        Z = sppy.diag(invNorms, storagetype="row")
        Y = Z.dot(Y)
        
        sigma = 0.1
        blocksize = 500
        
        numBlocks = int(ceil(Y.shape[0]/float(blocksize)))
        logging.debug("Number of blocks " + str(numBlocks))
        
        allRowInds = numpy.array([], numpy.int32)
        allColInds = numpy.array([], numpy.int32)
        allValues = numpy.array([], numpy.float)
        
        for i in range(numBlocks): 
            logging.debug("Iteration: " + str(i))
            
            endInd = min(Y.shape[0], (i+1)*blocksize)
                    
            tempY = Y.submatrix(i*blocksize, 0, endInd-i*blocksize, Y.shape[1])
            tempC = tempY.dot(Y.T)
            tempC = tempC.clip(sigma, 1.0)
    
            rowInds, colInds = tempC.nonzero()
            rowInds += i*blocksize
            values = tempC.values()
            
            allRowInds = numpy.r_[allRowInds, rowInds]
            allColInds = numpy.r_[allColInds, colInds]
            allValues = numpy.r_[allValues, values]
    
    
        coords = numpy.c_[allRowInds+1, allColInds+1, allValues] 
        comment = "%%MatrixMarket matrix coordinate real general\n"
        comment += "%\n"
        comment += str(Y.shape[0]) + " " + str(Y.shape[0]) + " " + str(allRowInds.shape[0])
        
        
        numpy.savetxt(authorAuthorFileName, coords, delimiter=" ", header=comment, comments="", fmt="%d %d %f")
        logging.debug("Saved matrix to " + authorAuthorFileName)
        logging.debug("Final size of C " + str(Y.shape[0]) + " with " + str(allRowInds.shape[0]) + " nonzeros")
    else: 
        logging.debug("File exists: " + authorAuthorFileName)    
    
    
    
writeAuthorDocMatrix()
writeAuthorAuthorMatrix()
#ProfileUtils.profile("writeAuthorAuthorMatrix()", globals(), locals())   