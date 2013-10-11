
"""
Run some experiments using the Epinions dataset 
"""
import os
import sys
import errno
import logging
import numpy
import argparse
import time 
from datetime import datetime
from wallhack.recommendexp.RecommendExpHelper import RecommendExpHelper
from wallhack.recommendexp.EpinionsDataset import EpinionsDataset

#Uncomment this for the final run 
if __debug__: 
    raise RuntimeError("Must run python with -O flag")

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise", under="ignore")

# Arguments related to the dataset
dataArgs = argparse.Namespace()
dataArgs.maxIter = 40 
#Set iterStartDate to None for all iterations 
#dataArgs.iterStartTimeStamp = None 
dataArgs.iterStartTimeStamp = time.mktime(datetime(2002,1,1).timetuple())

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = numpy.array(2**numpy.arange(2.5, 5.5, 0.5), numpy.int) 
defaultAlgoArgs.rhos = numpy.linspace(0.3, 0.1, 3) 
defaultAlgoArgs.folds = 4


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
dataArgs.extendedDirName += "EpinionsDataset"

# print args #
logging.info("Running on EpinionsDataset")
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

# data
generator = EpinionsDataset(maxIter=dataArgs.maxIter, iterStartTimeStamp=dataArgs.iterStartTimeStamp)

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
