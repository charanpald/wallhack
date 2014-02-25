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

X = X[0:10000, :]

(m, n) = X.shape
logging.debug("Size of X: " + str(X.shape))
logging.debug("Number of non zeros: " + str(X.nnz))

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = numpy.array([10, 20, 50, 100])
defaultAlgoArgs.rhos = numpy.flipud(numpy.logspace(-4, -1, 5)) 
defaultAlgoArgs.folds = 4

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
logging.info("Running on MendeleyCoauthors")
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

logging.info("Creating the exp-runner")
recommendExpHelper = RankingExpHelper(remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
recommendExpHelper.printAlgoArgs()
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(recommendExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

recommendExpHelper.runExperiment(X)
