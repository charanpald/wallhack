import numpy
import logging
import sys
import argparse 
import os
import errno
from sandbox.util.PathDefaults import PathDefaults 
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
import sppy 
import sppy.io

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
X = sppy.io.mmread(authorAuthorFileName, storagetype="row")
logging.debug("Read file: " + authorAuthorFileName)

#X = X[0:20000, :]

(m, n) = X.shape
logging.debug("Size of X: " + str(X.shape))
logging.debug("Number of non zeros: " + str(X.nnz))

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = 2**numpy.arange(3, 7)
defaultAlgoArgs.rhos = numpy.flipud(numpy.logspace(-7, -3, 5))
defaultAlgoArgs.lmbdasMlauc = 2.0**-numpy.arange(1, 12, 2)
#defaultAlgoArgs.lmbdasMlauc = numpy.array([0.0])
defaultAlgoArgs.maxIterations = m*50
defaultAlgoArgs.numRowSamples = 100
defaultAlgoArgs.numStepIterations = 1000
defaultAlgoArgs.numAucSamples = 10
defaultAlgoArgs.initialAlg = "rand"
defaultAlgoArgs.recordStep = defaultAlgoArgs.numStepIterations
defaultAlgoArgs.rate = "optimal"
defaultAlgoArgs.alpha = 0.1
defaultAlgoArgs.t0 = 0.0001
defaultAlgoArgs.folds = 3
defaultAlgoArgs.rhoMlauc = 0.0
defaultAlgoArgs.validationSize = 3
defaultAlgoArgs.u = 0.1 
defaultAlgoArgs.alphas = 2.0**-numpy.arange(-1, 2, 0.5)
defaultAlgoArgs.t0s = 2.0**-numpy.arange(6, 14, 1)

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "MendeleyCoauthors"

# print args #
logging.info("Running on " + dataArgs.extendedDirName)
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

logging.info("Creating the exp-runner")
rankingExpHelper = RankingExpHelper(remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
rankingExpHelper.printAlgoArgs()
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(rankingExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

rankingExpHelper.runExperiment(X)
