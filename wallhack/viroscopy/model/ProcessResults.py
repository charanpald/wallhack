import numpy
import logging
import sys 
import multiprocessing 
import matplotlib.pyplot as plt 
import os
from apgl.graph.GraphStatistics import GraphStatistics 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util 
from apgl.util.Latex import Latex 
from sandbox.predictors.ABCSMC import loadThetaArray 
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from sandbox.misc.GraphMatch import GraphMatch
from wallhack.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

processReal = False 
saveResults = True 

if processReal: 
    ind = 2
    N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize = HIVModelUtils.realABCParams(True)
    resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/theta" + str(ind) + "/"
    outputDir = resultsDir + "stats/"
    startDates, endDates, numRecordSteps, M, targetGraph = HIVModelUtils.realSimulationParams()
    startDate = startDates[ind]
    endDate = endDates[ind]
    endDate += (endDate-startDate)/10.0
    recordStep = (endDate-startDate)/float(numRecordSteps)
    
    
    realTheta, sigmaTheta, pertTheta = HIVModelUtils.estimatedRealTheta(ind)
    prefix = "Real"
else: 
    N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize = HIVModelUtils.toyABCParams()
    resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/theta/"
    outputDir = resultsDir + "stats/"
    startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
    endDate += HIVModelUtils.toyTestPeriod
    realTheta, sigmaTheta, pertTheta = HIVModelUtils.toyTheta()
    prefix = "Toy"

try: 
    os.mkdir(outputDir)
except: 
    logging.debug("Directory exists: " + outputDir) 

graphStats = GraphStatistics()
breakSize = targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size * breakScale
print(breakSize)
t = 0
maxT = numEpsilons
plotStyles = ['k-', 'kx-', 'k+-', 'k.-', 'k*-']

for i in range(maxT): 
    thetaArray, distArray = loadThetaArray(N, resultsDir, i)
    if thetaArray.shape[0] == N: 
        t = i   
    
logging.debug("Using population " + str(t))
#We plot some stats for the ideal simulated epidemic 
#and those epidemics found using ABC. 

def saveStats(args):
    i, theta = args 
    
    featureInds= numpy.ones(targetGraph.vlist.getNumFeatures(), numpy.bool)
    featureInds[HIVVertices.dobIndex] = False 
    featureInds[HIVVertices.infectionTimeIndex] = False 
    featureInds[HIVVertices.hiddenDegreeIndex] = False 
    featureInds[HIVVertices.stateIndex] = False 
    featureInds = numpy.arange(featureInds.shape[0])[featureInds]        
    graph = targetGraph.subgraph(targetGraph.removedIndsAt(startDate)) 
    graph.addVertices(M-graph.size)
    logging.debug("Created graph: " + str(graph))    
    
    matcher = GraphMatch(matchAlg, alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
    graphMetrics = HIVGraphMetrics2(targetGraph, breakSize, matcher, float(endDate))     
    times, infectedIndices, removedIndices, graph = HIVModelUtils.simulate(thetaArray[i], graph, startDate, endDate, recordStep, graphMetrics)
    times = numpy.arange(startDate, endDate+1, recordStep)
    vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats = HIVModelUtils.generateStatistics(graph, times)
    stats = times, vertexArray, infectedIndices, removedGraphStats, graphMetrics.objectives, graphMetrics.graphObjs, graphMetrics.labelObjs
    resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
    Util.savePickle(stats, resultsFileName)

if saveResults:
    thetaArray = loadThetaArray(N, resultsDir, t)[0]
    logging.debug(thetaArray)
    
    paramList = []
    
    for i in range(thetaArray.shape[0]): 
        paramList.append((i, thetaArray[i, :]))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())               
    resultIterator = pool.map(saveStats, paramList)  
    #resultIterator = map(saveStats, paramList)  
    pool.terminate()

    #Now save the statistics on the target graph 
    times = numpy.arange(startDate, endDate+1, recordStep)
    vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats = HIVModelUtils.generateStatistics(targetGraph, times)
    stats = vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats
    resultsFileName = outputDir + "IdealStats.pkl"
    Util.savePickle(stats, resultsFileName)
else:
    realTheta, sigmaTheta, purtTheta = HIVModelUtils.toyTheta()
    thetaArray, distArray = loadThetaArray(N, resultsDir, t)
    print(realTheta)
    print(thetaArray)    
    print(distArray)
    
    meanTable = numpy.c_[realTheta, thetaArray.mean(0)]
    stdTable = numpy.c_[sigmaTheta, thetaArray.std(0)]
    table = Latex.array2DToRows(meanTable, stdTable, precision=4)
    rowNames = ["$\\|\\mathcal{I}_0 \\|$", "$\\alpha$", "$\\gamma$", "$\\beta$", "$\\lambda$",  "$\\sigma$"]
    table = Latex.addRowNames(rowNames, table)
    print(table)

    resultsFileName = outputDir + "IdealStats.pkl"
    stats = Util.loadPickle(resultsFileName)  
    vertexArrayIdeal, idealInfectedIndices, idealRemovedIndices, idealContactGraphStats, idealRemovedGraphStats = stats 
    times = numpy.arange(startDate, endDate+1, recordStep)  
    
    graphStats = GraphStatistics()
    
    plotInd = 0 
    
    distsArr = []
    detectsArr = []
    infectsArr = []
    numComponentsArr = []
    randDetectsArr = []
    contactTracingArr = []
    maxCompSizesArr = []
    numEdgesArr = []

    for i in range(thetaArray.shape[0]): 
        plotInd = 0
        resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
        stats = Util.loadPickle(resultsFileName)
        
        times, vertexArray, infectedIndices, removedGraphStats, dists, graphDists, labelDists = stats 

        detectsArr.append(vertexArray[:, 0])
        randDetectsArr.append(vertexArray[:, 5])
        contactTracingArr.append(vertexArray[:, 6])
        numComponentsArr.append(removedGraphStats[:, graphStats.numComponentsIndex])
        distsArr.append(dists)        
        numInfects = [len(x) for x in infectedIndices]
        infectsArr.append(numInfects)
        maxCompSizesArr.append(removedGraphStats[:, graphStats.maxComponentSizeIndex])
        numEdgesArr.append(removedGraphStats[:, graphStats.numEdgesIndex])

    times = times - numpy.min(times)

    contactTracingArr = numpy.array(contactTracingArr)
    meanContactDetectsArr = numpy.mean(contactTracingArr, 0)
    stdContactDetectsArr = numpy.std(contactTracingArr, 0)
    plt.figure(plotInd)
    plt.errorbar(times, meanContactDetectsArr, yerr=stdContactDetectsArr, label="est. CT detections") 
    plt.plot(times, vertexArrayIdeal[:, 6], "r", label="CT detections")

    meanRandDetects = numpy.array(randDetectsArr).mean(0)
    stdRandDetects = numpy.array(randDetectsArr).std(0)
    plt.errorbar(times, meanRandDetects, yerr=stdRandDetects, label="est. rand detections") 
    plt.xlabel("time (days)")
    plt.ylabel("detections")
    plt.plot(times, vertexArrayIdeal[:, 5], "k", label="rand detections")
    if not processReal: 
        lims = plt.xlim()
        plt.xlim([0, lims[1]]) 
    plt.legend(loc="upper left")
    plt.savefig(outputDir + prefix + "CTRandDetects.eps")
    plotInd += 1
    

    plt.figure(plotInd)    
    if not processReal: 
        infectsArr = numpy.array(infectsArr)
        meanInfectsArr = numpy.mean(infectsArr, 0)
        stdInfectsArr = numpy.std(infectsArr, 0)
        numInfects = [len(x) for x in idealInfectedIndices]
        plt.errorbar(times, meanInfectsArr, yerr=stdInfectsArr, label="est. infectives") 
        plt.plot(times, numInfects, "r", label="infectives")

    meanDetects = numpy.array(detectsArr).mean(0)
    stdDetects = numpy.array(detectsArr).std(0)
    plt.errorbar(times, meanDetects, yerr=stdDetects, label="est. detections") 
    plt.xlabel("time (days)")
    plt.plot(times, vertexArrayIdeal[:, 0], "k", label="detections")
    if not processReal:
        plt.ylabel("infectives/detections")
        lims = plt.xlim()
        plt.xlim([0, lims[1]]) 
        filename = outputDir + prefix + "InfectDetects.eps"
    else: 
        plt.ylabel("detections")
        filename = outputDir + prefix + "Detects" + str(ind) + ".eps"

    plt.legend(loc="lower right")
    plt.savefig(filename)
    plotInd += 1    
    
    numComponentsArr = numpy.array(numComponentsArr)
    meanNumComponents = numpy.mean(numComponentsArr, 0)
    stdNumComponents = numpy.std(numComponentsArr, 0)
    plt.figure(plotInd)
    plt.errorbar(times, meanNumComponents, yerr=stdNumComponents) 
    plt.xlabel("time (days)")
    plt.ylabel("num components")
    plt.plot(times, idealRemovedGraphStats[:, graphStats.numComponentsIndex], "r")
    plotInd += 1
    
    maxCompSizesArr = numpy.array(maxCompSizesArr)
    meanMaxCompSizesArr = numpy.mean(maxCompSizesArr, 0)
    stdMaxCompSizesArr = numpy.std(maxCompSizesArr, 0)
    plt.figure(plotInd)
    plt.errorbar(times, meanMaxCompSizesArr, yerr=stdMaxCompSizesArr) 
    plt.xlabel("time (days)")
    plt.ylabel("max component size")
    plt.plot(times, idealRemovedGraphStats[:, graphStats.maxComponentSizeIndex], "r")
    plotInd += 1
    
    numEdgesArr = numpy.array(numEdgesArr)
    meanNumEdgesArr = numpy.mean(numEdgesArr, 0)
    stdNumEdgesArr = numpy.std(numEdgesArr, 0)
    plt.figure(plotInd)
    plt.errorbar(times, meanNumEdgesArr, yerr=stdNumEdgesArr) 
    plt.xlabel("time (days)")
    plt.ylabel("number of edges")
    plt.plot(times, idealRemovedGraphStats[:, graphStats.numEdgesIndex], "r")
    plotInd += 1
    
    """
    meanDists = numpy.array(distsArr).mean(0)
    stdDists = numpy.array(distsArr).std(0)
    plt.figure(plotInd)
    plt.errorbar(times[1:], meanDists, yerr=stdDists) 
    plt.xlabel("time (days)")
    plt.ylabel("distances")
    plotInd += 1
    """
    
    plt.show()
