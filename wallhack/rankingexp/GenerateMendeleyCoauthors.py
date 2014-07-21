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
import os.path
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def writeAuthorXMatrix(inputFileName, authorIndexerFilename, authorXFileName, reverse=False): 
    
    if not os.path.isfile(authorXFileName): 
        fileObj = open(inputFileName)
        
        authorIndex = IdIndexer()
        docIndex = IdIndexer()
        scores = array.array("i")
        
        for i, line in enumerate(fileObj):
            if i % 500000 == 0: 
                logging.debug(i)
            vals = line.split()
            #logging.debug(vals[0], vals[1], vals[2])
            
            if reverse: 
                authorIndex.append(vals[1])
                docIndex.append(vals[0])
            else: 
                authorIndex.append(vals[0])
                docIndex.append(vals[1])
                
            score = int(vals[2])
            scores.append(int(score))
        
        rowInds = numpy.array(authorIndex.getArray())
        colInds = numpy.array(docIndex.getArray())
        
        Y = scipy.sparse.csr_matrix((scores, (rowInds, colInds)))
            
        authorIndexerFile = open(authorIndexerFilename, "wb")
        pickle.dump(authorIndex, authorIndexerFile)
        authorIndexerFile.close()
        scipy.io.mmwrite(authorXFileName, Y)
        logging.debug("Saved matrix to " + authorXFileName)
    else: 
        logging.debug("File exists: " + authorXFileName)
    
def writeAuthorAuthorMatrix(authorXFileName, authorAuthorFileName, sigma=0.05): 
    
    if not os.path.isfile(authorAuthorFileName): 
        Y = sppy.io.mmread(authorXFileName, storagetype="row")
        logging.debug("Read file: " + authorXFileName)
        logging.debug("Size of input: " + str(Y.shape) + " with " + str(Y.nnz) + " nonzeros")
    
        Y = sppy.csarray(Y, dtype=numpy.float, storagetype="row")
        #Y = Y[0:1000, :]
        
        invNorms = 1/numpy.sqrt((Y.power(2).sum(1)))
        Z = sppy.diag(invNorms, storagetype="row")
        Y = Z.dot(Y)
        
        blocksize = 100
        
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
            
            logging.debug(allValues.shape)
    
    
        coords = numpy.c_[allRowInds+1, allColInds+1, allValues] 
        comment = "%%MatrixMarket matrix coordinate real general\n"
        comment += "%\n"
        comment += str(Y.shape[0]) + " " + str(Y.shape[0]) + " " + str(allRowInds.shape[0])
        
        
        numpy.savetxt(authorAuthorFileName, coords, delimiter=" ", header=comment, comments="", fmt="%d %d %f")
        logging.debug("Saved matrix to " + authorAuthorFileName)
        logging.debug("Final size of C " + str(Y.shape[0]) + " with " + str(allRowInds.shape[0]) + " nonzeros")
    else: 
        logging.debug("File exists: " + authorAuthorFileName)    
    
dataDir = PathDefaults.getDataDir()

#Write out author-author from documents 
inputFileName = dataDir + "reference/author_document_count"
authorXFileName = dataDir + "reference/authorDocumentMatrix.mtx"  
authorIndexerFilename = dataDir + "reference/authorIndexerDoc.pkl"    
authorAuthorFileName = dataDir + "reference/authorAuthorDocMatrix.mtx" 
writeAuthorXMatrix(inputFileName, authorIndexerFilename, authorXFileName, reverse=True)
writeAuthorAuthorMatrix(authorXFileName, authorAuthorFileName)

#Write out author-author from keywords
inputFileName = dataDir + "reference/author_keyword_count"
authorXFileName = dataDir + "reference/authorKeywordMatrix.mtx"    
authorIndexerFilename = dataDir + "reference/authorIndexerKeyword.pkl"    
authorAuthorFileName = dataDir + "reference/authorAuthorKeywordMatrix.mtx" 
writeAuthorXMatrix(inputFileName, authorIndexerFilename, authorXFileName, reverse=True)
writeAuthorAuthorMatrix(authorXFileName, authorAuthorFileName, sigma=0.6)