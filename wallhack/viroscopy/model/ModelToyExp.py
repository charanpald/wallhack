"""
A script to estimate the HIV epidemic model parameters using ABC for the toy data.
"""

from sandbox.util.PathDefaults import PathDefaults
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
from wallhack.viroscopy.model.HIVABCParameters import HIVABCParameters
from sandbox.predictors.ABCSMC import ABCSMC
import os
import logging
import sys
import numpy
import multiprocessing

assert False, "Must run with -O flag"

if len(sys.argv) > 1:
    numProcesses = int(sys.argv[1])
else: 
    numProcesses = multiprocessing.cpu_count()

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.debug("Number of processes: " + str(numProcesses))
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale  = HIVModelUtils.toyABCParams()

logging.debug("Total time of simulation is " + str(endDate-startDate))
logging.debug("Posterior sample size " + str(N))

epsilonArray = numpy.ones(numEpsilons)*epsilon   
breakSize = (targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size - targetGraph.subgraph(targetGraph.removedIndsAt(startDate)).size)  * breakScale
logging.debug("Largest acceptable graph is " + str(breakSize))

def createModel(t):
    """
    The parameter t is the particle index. 
    """
    return HIVModelUtils.createModel(targetGraph, startDate, endDate, recordStep, M, matchAlpha, breakSize, matchAlg)

meanTheta, sigmaTheta, pertTheta = HIVModelUtils.toyTheta()
abcParams = HIVABCParameters(meanTheta, sigmaTheta, pertTheta)
thetaDir = resultsDir + "theta/"

if not os.path.exists(thetaDir): 
    os.mkdir(thetaDir)

logging.debug((meanTheta, sigmaTheta))

abcSMC = ABCSMC(epsilonArray, createModel, abcParams, thetaDir, True, minEpsilon=minEpsilon)
abcSMC.setPosteriorSampleSize(N)
abcSMC.batchSize = batchSize
abcSMC.maxRuns = abcMaxRuns
abcSMC.setNumProcesses(numProcesses)
abcSMC.pertScale = pertScale
thetasArray = abcSMC.run()

meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)
logging.debug(thetasArray)
logging.debug("meanTheta=" + str(meanTheta))
logging.debug("stdTheta=" + str(stdTheta))
logging.debug("realTheta=" + str(HIVModelUtils.toyTheta()[0]))
logging.debug("Final epsilon array: " + str(abcSMC.epsilonArray))
logging.debug("Number of ABC runs: " + str(abcSMC.numRuns))
logging.debug("All done!")
