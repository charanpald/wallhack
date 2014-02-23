import numpy 
import scipy.io
import scipy.sparse.linalg 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt  
from sandbox.util.PathDefaults import PathDefaults 

"""
Let's look at the eigenvalues of the laplacians and their k-cores 
"""

dataDir = PathDefaults.getDataDir() + "kcore/"

matDict = scipy.io.loadmat(dataDir + "network_1_cores_adjacency_laplacian.mat")
k = 50

for i in range(1, 6): 
    L = matDict["L_core0" + str(i)]
    print(L.shape)
    
    u, V = scipy.sparse.linalg.eigs(L, k=k, which="SM")
    
    u = numpy.sort(u)
    #plt.figure(i)
    plt.plot(numpy.arange(k, dtype=numpy.float)/L.shape[0], u, label="core " + str(i))
    
plt.legend()
plt.show()
