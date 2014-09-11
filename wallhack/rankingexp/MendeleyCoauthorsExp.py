import numpy
import logging
import sys
import argparse 
from sandbox.util.Sampling import Sampling 
from wallhack.rankingexp.DatasetUtils import DatasetUtils 
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = numpy.array([64])
defaultAlgoArgs.parallelSGD = True
defaultAlgoArgs.recordFolds = 1

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.dataset = "Doc"
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
dataParser.add_argument("--dataset", type=str, help="The dataset to use: either Doc or Keyword (default: %(default)s)", default=dataParser.dataset)
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

defaultAlgoArgs.u = 0.1 
defaultAlgoArgs.maxIterations = 30 
defaultAlgoArgs.numRowSamples = 10 

# print args #
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

logging.info("Creating the exp-runner")

#Load/create the dataset - sample at most a million nnzs
X = DatasetUtils.mendeley(dataset=dataArgs.dataset)
X, userInds = Sampling.sampleUsers2(X, 10**6)
m, n = X.shape

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "MendeleyCoauthors" + dataParser.dataset

rankingExpHelper = RankingExpHelper(remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
rankingExpHelper.printAlgoArgs()
rankingExpHelper.runExperiment(X)
