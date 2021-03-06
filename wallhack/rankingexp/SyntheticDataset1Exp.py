import numpy
import logging
import sys
import argparse 
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.Util import Util 

Util.setupScript()

#Create a low rank matrix  
X, U, V = DatasetUtils.syntheticDataset1(u=0.2, sd=0.2)
m, n = X.shape
u = 0.1
w = 1-u

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.u = 5/float(n) 
#defaultAlgoArgs.validationUsers = 0.0
defaultAlgoArgs.ks = numpy.array([8])

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "SyntheticDataset1"

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
rankingExpHelper.runExperiment(X)