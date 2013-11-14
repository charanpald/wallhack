"""
Perform cross validation using TreeRank
"""
import os
import numpy
import sys
import logging
import multiprocessing
import datetime
import gc 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from sandbox.ranking.TreeRank import TreeRank
from sandbox.ranking.TreeRankForest import TreeRankForest
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from socket import gethostname

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.debug("Running from machine " + str(gethostname()))
numpy.random.seed(21)

dataDir = PathDefaults.getDataDir() +  "metabolomic/"
X, X2, Xs, XOpls, YList, ages, df = MetabolomicsUtils.loadData()

mode = "cpd"
level = 10
XwDb4 = MetabolomicsUtils.getWaveletFeatures(X, 'db4', level, mode)
XwDb8 = MetabolomicsUtils.getWaveletFeatures(X, 'db8', level, mode)
XwHaar = MetabolomicsUtils.getWaveletFeatures(X, 'haar', level, mode)

#Filter the wavelets
Ns = [10, 25, 50, 75, 100]
dataList = []

for i in range(len(Ns)):
    N = Ns[i]
    XwDb4F, inds = MetabolomicsUtils.filterWavelet(XwDb4, N)
    dataList.append((XwDb4F[:, inds], "Db4-" + str(N)))

    XwDb8F, inds = MetabolomicsUtils.filterWavelet(XwDb8, N)
    dataList.append((XwDb8F[:, inds], "Db8-" + str(N)))

    XwHaarF, inds = MetabolomicsUtils.filterWavelet(XwHaar, N)
    dataList.append((XwHaarF[:, inds], "Haar-" + str(N)))

dataList.extend([(Xs, "raw_std"), (XwDb4, "Db4"), (XwDb8, "Db8"), (XwHaar, "Haar"), (X2, "log"), (XOpls, "opls")])

#Data for functional TreeRank
dataListF = [(XwDb4, "Db4"), (XwDb8, "Db8"), (XwHaar, "Haar")]
dataListPCA = ([(Xs, "raw_std"), (XwDb4, "Db4"), (XwDb8, "Db8"), (XwHaar, "Haar"), (X2, "log"), (XOpls, "opls")])

lock = multiprocessing.Lock()

numpy.random.seed(datetime.datetime.now().microsecond)
#numpy.random.seed(21)
permInds = numpy.random.permutation(len(dataList))
permIndsF = numpy.random.permutation(len(dataListF))
permIndsPCA = numpy.random.permutation(len(dataListPCA))
numpy.random.seed(21)

try:
    for ind in permInds:
        MetabolomicsExpRunner(YList, dataList[ind][0], dataList[ind][1], ages, args=(lock,)).run()
        
    for ind in permIndsF:
        MetabolomicsExpRunner(YList, dataListF[ind][0], dataListF[ind][1], ages, args=(lock,)).runF()

    for ind in permIndsPCA:
        MetabolomicsExpRunner(YList, dataListPCA[ind][0], dataListPCA[ind][1], ages, args=(lock,)).runPCA()

    logging.info("All done - see you around!")
except Exception as err:
    print(err)
    raise 
