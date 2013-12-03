"""
A script to estimate the HIV epidemic model parameters using ABC for real data.
"""
from apgl.util.PathDefaults import PathDefaults
from wallhack.viroscopy.model.HIVABCParameters import HIVABCParameters
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
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

if len(sys.argv) > 2:
    i = int(sys.argv[2])
else: 
    i = 0 

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.debug("Number of processes: " + str(numProcesses))
logging.debug("Epidemic period index " + str(i))
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/" 
startDate, endDate, recordStep, M, targetGraph, numInds = HIVModelUtils.realSimulationParams(ind=i)
N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale = HIVModelUtils.realABCParams(i)

logging.debug("Posterior sample size " + str(N))
logging.debug("Matching algorithm " + str(matchAlg))

logging.debug("="*10 + "Starting new simulation batch with index " + str(i) + "="*10) 
logging.debug("Total time of simulation is " + str(endDate-startDate))    

breakSize = (targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size - targetGraph.subgraph(targetGraph.removedIndsAt(startDate)).size)  * breakScale
logging.debug("Largest acceptable graph is " + str(breakSize))

def createModel(t):
    """
    The parameter t is the particle index. 
    """
    return HIVModelUtils.createModel(targetGraph, startDate, endDate, recordStep, M, matchAlpha, breakSize, matchAlg)

meanTheta, stdTheta, pertTheta = HIVModelUtils.estimatedRealTheta(i)

logging.debug("Using mean theta of " + str(meanTheta))
logging.debug("Using std theta of " + str(stdTheta))
logging.debug("Using perturbation std theta of " + str(pertTheta))
    
abcParams = HIVABCParameters(meanTheta, stdTheta, pertTheta)
thetaDir = resultsDir + "theta" + str(i) + "/"

if not os.path.exists(thetaDir): 
    os.mkdir(thetaDir)

#Now get something more precise 
epsilonArray = numpy.ones(numEpsilons)*epsilon    
abcSMC = ABCSMC(epsilonArray, createModel, abcParams, thetaDir, True, minEpsilon=minEpsilon, thetaUniformChoice=False)
abcSMC.setPosteriorSampleSize(N)
abcSMC.setNumProcesses(numProcesses)
abcSMC.batchSize = batchSize
abcSMC.maxRuns = abcMaxRuns
abcSMC.pertScale = pertScale
thetasArray = abcSMC.run()

meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)
logging.debug(thetasArray)
logging.debug("meanTheta=" + str(meanTheta))
logging.debug("stdTheta=" + str(stdTheta))
logging.debug("Final epsilon array: " + str(abcSMC.epsilonArray))
logging.debug("Number of ABC runs: " + str(abcSMC.numRuns))
logging.debug("All done!")
