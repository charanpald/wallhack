import array
import numpy
import scipy.sparse
import scipy.io
from apgl.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer
from math import ceil 
import sppy 
import sppy.io

def writeAuthorDocMatrix(): 
    fileName = PathDefaults.getDataDir() + "reference/author_document_count"
    
    fileObj = open(fileName)
    
    authorIndex = IdIndexer()
    docIndex = IdIndexer()
    scores = array.array("i")
    
    for i, line in enumerate(fileObj):
        if i % 500000 == 0: 
            print(i)
        vals = line.split()
        #print(vals[0], vals[1], vals[2])
        
        authorIndex.append(vals[1])
        docIndex.append(vals[0])
        scores.append(int(vals[2]))
    
    
    rowInds = numpy.array(authorIndex.getArray())
    colInds = numpy.array(docIndex.getArray())
    
    Y = scipy.sparse.csr_matrix((scores, (rowInds, colInds)))
    
    print(Y.shape, Y.nnz)
    
    outFileName = PathDefaults.getDataDir() + "reference/authorDocumentMatrix.mtx" 
    scipy.io.mmwrite(outFileName, Y)
    print("Saved matrix to " + outFileName)
    
def writeAuthorAuthorMatrix(): 
    matrixFileName = PathDefaults.getDataDir() + "reference/authorDocumentMatrix.mtx"
    Y = scipy.io.mmread(matrixFileName)
    print("Read file: " + matrixFileName)
    Y = Y.tocsr()
    
    Y = sppy.csarray(Y)
    Y = Y[0:53, :]
    
    invNorms = 1/(Y.power(2).sum(1))
    Z = sppy.diag(invNorms)
    Y = Z.dot(Y)
    
    print(Y.shape)
    
    sigma = 0.5
    blocksize = 10
    
    C = sppy.csarray((Y.shape[0], Y.shape[0]), storagetype="row")
    numBlocks = int(ceil(C.shape[0]/blocksize))
    print(numBlocks)
    
    for i in range(numBlocks): 
        print(i)
        endInd = min(Y.shape[0], (i+1)*blocksize)
        tempC = Y[i*blocksize:endInd, :].dot(Y.T)
        print("here")
        tempC.clip(sigma, 1.0)
        print("here2")
        
        rowInds, colInds = tempC.nonzero()
        print("here3")
        rowInds += i*blocksize
        print("here4")
        
        C.put(tempC.values(), rowInds, colInds)
    
    outFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
    sppy.io.mmwrite(outFileName, C)
    print("Saved matrix to " + outFileName)
    
#writeAuthorDocMatrix()
writeAuthorAuthorMatrix()        
        