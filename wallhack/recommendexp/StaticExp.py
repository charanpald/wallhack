
"""
Run some experiments using a static dataset 
"""
import os
import sys
import errno
import logging
import numpy
import argparse
import time 
from wallhack.recommendexp.RecommendExpHelper import RecommendExpHelper
from wallhack.recommendexp.Static2IdValDataset import Static2IdValDataset 
from apgl.util.PathDefaults import PathDefaults

#Uncomment this for the final run 
#if __debug__: 
#    raise RuntimeError("Must run python with -O flag")

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise", under="ignore")

# Arguments related to the dataset
dataArgs = argparse.Namespace()


# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = numpy.array([5, 10, 20, 30]) 
defaultAlgoArgs.rhos = numpy.flipud(numpy.arange(0.05, 0.45, 0.05))
defaultAlgoArgs.folds = 5


# init (reading/writting command line arguments)
# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RecommendExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "AuthorDocumentDataset"

# print args #
logging.info("Running on AuthorDocumentDataset")
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

# data
dataFilename = PathDefaults.getDataDir() + "reference/author_document_count" 
generator = Static2IdValDataset(dataFilename)

# run
logging.info("Creating the exp-runner")
recommendExpHelper = RecommendExpHelper(generator.getTrainIteratorFunc, generator.getTestIteratorFunc, remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
recommendExpHelper.printAlgoArgs()
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(recommendExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

recommendExpHelper.runExperiment()
