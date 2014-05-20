import numpy
import logging
import sys
import argparse 
import os
import errno
import sppy 
import array
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.IdIndexer import IdIndexer 
from sandbox.util.SparseUtils import SparseUtils 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = 2**numpy.arange(3, 7)

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

#Create/load a low rank matrix 
matrixFileName = PathDefaults.getDataDir() + "flixster/Ratings.timed.txt" 
matrixFile = open(matrixFileName)
matrixFile.readline()
userIndexer = IdIndexer("i")
movieIndexer = IdIndexer("i")

ratings = array.array("f")
logging.debug("Loading ratings from " + matrixFileName)

for i, line in enumerate(matrixFile):
    if i % 1000000 == 0: 
        print("Iteration: " + str(i))
    vals = line.split()
    
    userIndexer.append(vals[0])
    movieIndexer.append(vals[1])
    ratings.append(float(vals[2]))

rowInds = userIndexer.getArray()
colInds = movieIndexer.getArray()
ratings = numpy.array(ratings)

X = sppy.csarray((len(userIndexer.getIdDict()), len(movieIndexer.getIdDict())), storagetype="row", dtype=numpy.int)
X.put(numpy.array(ratings>3, numpy.int), numpy.array(rowInds, numpy.int32), numpy.array(colInds, numpy.int32), init=True)
X = SparseUtils.pruneMatrix(X, minNnzRows=10, minNnzCols=10)
logging.debug("Read file: " + matrixFileName)
logging.debug("Shape of data: " + str(X.shape))
logging.debug("Number of non zeros " + str(X.nnz))
(m, n) = X.shape

defaultAlgoArgs.u = 5/float(n) 

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "Flixster"

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