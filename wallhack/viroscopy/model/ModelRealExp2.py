"""
A script to estimate the HIV epidemic model parameters using 
Simulated Annealing. 
"""
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVABCParameters import HIVABCParameters
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
from wallhack.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from sandbox.misc.GraphMatch import GraphMatch
from sandbox.predictors.ABCSMC import ABCSMC, loadThetaArray
import os
import logging
import sys
import numpy
import multiprocessing
import scipy.optimize 

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

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/" 
startDates, endDates, numRecordSteps, M, targetGraph = HIVModelUtils.realSimulationParams()
N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize = HIVModelUtils.realABCParams()

logging.debug("Posterior sample size " + str(N))

alpha = 2
zeroVal = 0.9

for i, endDate in enumerate(endDates): 
    startDate = startDates[i]
    logging.debug("="*10 + "Starting new simulation batch with index " + str(i) + "="*10) 
    logging.debug("Total time of simulation is " + str(endDate-startDate))    
    
    breakSize = targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size * breakScale
    logging.debug("Largest acceptable graph is " + str(breakSize))
    
    def simulateModel(theta):
        """
        The parameter t is the particle index. 
        """
        logging.debug("theta=" + str(theta))
 
        #We start with the observed graph at the start date 
        graph = targetGraph.subgraph(targetGraph.removedIndsAt(startDate)) 
        graph.addVertices(M-graph.size)

        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
        
        featureInds = numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
        featureInds[HIVVertices.dobIndex] = False 
        featureInds[HIVVertices.infectionTimeIndex] = False 
        featureInds[HIVVertices.hiddenDegreeIndex] = False 
        featureInds[HIVVertices.stateIndex] = False
        featureInds = numpy.arange(featureInds.shape[0])[featureInds]
        matcher = GraphMatch(matchAlg, alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
        graphMetrics = HIVGraphMetrics2(targetGraph, breakSize, matcher, float(endDate))
        
        recordStep = (endDate-startDate)/float(numRecordSteps)
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
        model.setRecordStep(recordStep)
        model.setParams(theta)
        
        model.simulate() 
    
        objective = model.objective()
        return objective
        
    def optimise(args): 
        seed = args 
        numpy.random.seed(seed)
        lowerTheta = numpy.array([0, 0.5, 0, 0, 0, 0])
        upperTheta = numpy.array([500, 1.0, 0.5, 0.5, 0.5, 0.2])
        initialTheta = numpy.array([100, 0.9, 0.05, 0.001, 0.1, 0.005])
        results = scipy.optimize.anneal(simulateModel, x0=initialTheta, full_output=True, maxiter=5, maxaccept=10, lower=lowerTheta, upper=upperTheta)
        
        return results 
        
    thetaDir = resultsDir + "thetaSA" + str(i) + "/"
    
    if not os.path.exists(thetaDir): 
        os.mkdir(thetaDir)
    
    paramList = numpy.arange(10) 
    pool = multiprocessing.Pool(processes=numProcesses)               
    resultsIterator = pool.map(optimise, paramList)     
    #resultsIterator = map(runModel, paramList)     

    for result in resultsIterator: 
        theta = result[0]
        objective = result[1]
        logging.debug("Accepting theta=" + str(result[0]))
        logging.debug("Objective=" + str(result[1]))
        logging.debug("Number of iterations: " + str(result[4]))        
        
        currentTheta = loadThetaArray(N, thetaDir, 0)[0].tolist()
        fileName = thetaDir + "theta_" + str(len(currentTheta)) + ".npz" 
        try:
           with open(fileName, "w") as fileObj:
               numpy.savez(fileObj, theta, numpy.array(objective))
        except IOError:
           logging.debug("File IOError (probably a collision) occured with " + fileName)
    
    pool.terminate()

logging.debug("All done!")
