#Find the wavelet reconstruction of the data
import logging
import numpy
import sys
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from sandbox.util.PathDefaults import PathDefaults
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

mode = "cpd"
level = 10
XwDb4 = MetabolomicsUtils.getWaveletFeatures(X, 'db4', level, mode)
XwDb8 = MetabolomicsUtils.getWaveletFeatures(X, 'db8', level, mode)
XwHaar = MetabolomicsUtils.getWaveletFeatures(X, 'haar', level, mode)

plotStyles = ['k-', 'k--', 'k-.', 'k:', 'k.']

#Plot the correlation of the raw spectrum above x percent
Xr = numpy.random.rand(XStd.shape[0], XStd.shape[1])
datasets = [(Xr, "random"), (XStd, "raw"), (XwHaar, "Haar"), (XwDb4, "Db4"), (XwDb8, "Db8")]

corLims = numpy.arange(0, 1.01, 0.01)

for j, dataset in enumerate(datasets):
    X = Standardiser().standardiseArray(dataset[0])
    C = X.T.dot(X)

    w, V = numpy.linalg.eig(dataset[0].T.dot(dataset[0]))
    w = numpy.flipud(numpy.sort(w))

    correlations = numpy.zeros(corLims.shape[0])
    upperC = C[numpy.tril_indices(C.shape[0])]

    for i in range(corLims.shape[0]):
        correlations[i] = numpy.sum(numpy.abs(upperC) >= corLims[i])/float(upperC.size)

    plt.plot(corLims, correlations, plotStyles[j], label=dataset[1])

plt.xlabel("correlation > x")
plt.ylabel("probability")
plt.legend()
plt.show()

