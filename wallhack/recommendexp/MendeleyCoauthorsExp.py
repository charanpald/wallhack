import array
import numpy
import scipy.sparse
import scipy.io
from apgl.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer


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
    
    sigma = 0.5
    C = scipy.sparse.lil_matrix(Y.shape)
    
    for i in range(C.shape[0]): 
        if i % 10000 == 0: 
            print(i)
        for j in range(i, C.shape[0]): 
            if Y[i, :].dot(Y[j, :].T)[0,0] > sigma: 
                C[i, j] = 1
                C[j, i] = 1

    
    outFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
    scipy.io.mmwrite(outFileName, C)
    print("Saved matrix to " + outFileName)
    
        
writeAuthorAuthorMatrix()        
        