import numpy
import logging
import sys
import argparse 
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
from wallhack.rankingexp.DatasetUtils import DatasetUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

#Load/create the dataset 
X = DatasetUtils.movieLens()
(m, n) = X.shape

defaultAlgoArgs.u = 0.1
defaultAlgoArgs.validationUsers = 0.0

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "MovieLens"

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