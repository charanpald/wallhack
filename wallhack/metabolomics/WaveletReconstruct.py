#Find the wavelet reconstruction of the data 
import logging
import numpy
import sys
import pywt 
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from apgl.util.PathDefaults import PathDefaults
from socket import gethostname
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from sandbox.data.Standardiser import Standardiser

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.debug("Running from machine " + str(gethostname()))
numpy.random.seed(21)
numpy.set_printoptions(linewidth=160, precision=3, suppress=True)

dataDir = PathDefaults.getDataDir() +  "metabolomic/"
metaUtils = MetabolomicsUtils() 
X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()

waveletStr = 'db4'
mode = "cpd"
maxLevel = 10
errors = numpy.zeros(maxLevel)
numFeatures = numpy.zeros(maxLevel)

level = 10 
waveletStrs = ["haar", "db4", "db8"]
plt.figure(0)

C = XStd.T.dot(XStd)
w, V = numpy.linalg.eigh(C)
w = numpy.flipud(numpy.sort(w))

numEigs = 100
plt.plot(numpy.arange(numEigs), w[0:numEigs], "k")
plt.xlabel("eigenvalue rank")
plt.ylabel("eigenvalue")
print(numpy.sum(w[0:numEigs])/numpy.sum(w))

#Now try some filtering and plot N versus reconstruction error
Ns = range(0, 700, 50)
waveletStrs = ['haar', 'db4', 'db8']
waveletStrs2 = ['Haar', 'Db4', 'Db8']
errors = numpy.zeros((len(waveletStrs), len(Ns)))
mode = "cpd"

standardiser = Standardiser()
#X = standardiser.centreArray(X)

for i in range(len(waveletStrs)):
    print(i)
    waveletStr = waveletStrs[i]
    Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
    C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)

    for j in range(len(Ns)):
        N = Ns[j]
        Xw2, inds = MetabolomicsUtils.filterWavelet(Xw, N)
        X2 = MetabolomicsUtils.reconstructSignal(X, Xw2, waveletStr, mode, C)

        errors[i, j] = numpy.linalg.norm(X - X2)

#Plot example wavelet after filtering 
waveletStr = "haar"
N = 100
Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
Xw2, inds = MetabolomicsUtils.filterWavelet(Xw, N)
X2 = MetabolomicsUtils.reconstructSignal(X, Xw2, waveletStr, mode, C)

plt.figure(3)
plt.plot(range(X.shape[1]), X[0, :])
plt.plot(range(X.shape[1]), X2[0, :])
plt.xlabel("feature no.")
plt.ylabel("value")

plt.figure(4)
for i in range(errors.shape[0]):
    plt.plot(Ns, errors[i, :], label=waveletStrs2[i])
    plt.xlabel("N")
    plt.ylabel("error")

print(errors)

plt.legend()
plt.show()