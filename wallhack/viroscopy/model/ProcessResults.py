import numpy
import logging
import sys 
import multiprocessing 
import os
from apgl.graph.GraphStatistics import GraphStatistics 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util 
from apgl.util.Latex import Latex 
from apgl.util.FileLock import FileLock
from sandbox.predictors.ABCSMC import loadThetaArray 
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from sandbox.misc.GraphMatch import GraphMatch
from wallhack.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2

assert False, "Must run with -O flag"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

processReal = True 
saveResults = True 

def loadParams(ind): 
    if processReal: 
        resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/theta" + str(ind) + "/"
        outputDir = resultsDir + "stats/"
        
        N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize = HIVModelUtils.realABCParams(True)
        startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.realSimulationParams(test=True, ind=ind)
        realTheta, sigmaTheta, pertTheta = HIVModelUtils.estimatedRealTheta(ind)
        prefix = "Real"
        numInds = 3
    else: 
        resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/theta/"
        outputDir = resultsDir + "stats/"        
        
        N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize = HIVModelUtils.toyABCParams()
        startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams(test=True)
        realTheta, sigmaTheta, pertTheta = HIVModelUtils.toyTheta()
        prefix = "Toy"
        numInds = 1

    breakSize = targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size * breakScale        
        
    return N, resultsDir, outputDir, recordStep, startDate, endDate, prefix, targetGraph, breakSize, numEpsilons, M, matchAlpha, matchAlg, numInds

def saveStats(args):    
    i, theta = args 
    
    resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
    lock = FileLock(resultsFileName)
    
    if not lock.fileExists() and not lock.isLocked():    
        lock.lock()
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
        
        Util.savePickle(stats, resultsFileName)
        lock.unlock()
    else: 
        logging.debug("Results already computed: " + str(resultsFileName))

N, resultsDir, outputDir, recordStep, startDate, endDate, prefix, targetGraph, breakSize, numEpsilons, M, matchAlpha, matchAlg, numInds = loadParams(0)

if saveResults:
    for ind in range(numInds):
        logging.debug("Record step: " + str(recordStep))
        logging.debug("Start date: " + str(startDate))
        logging.debug("End date: " + str(endDate))
        logging.debug("End date - start date: " + str(endDate - startDate))
        
        N, resultsDir, outputDir, recordStep, startDate, endDate, prefix, targetGraph, breakSize, numEpsilons, M, matchAlpha, matchAlg, numInds = loadParams(ind)        
        
        t = 0
        for i in range(numEpsilons): 
            thetaArray, distArray = loadThetaArray(N, resultsDir, i)
            if thetaArray.shape[0] == N: 
                t = i   
            
        logging.debug("Using population " + str(t))        
        
        try: 
            os.mkdir(outputDir)
        except: 
            logging.debug("Directory exists: " + outputDir)     
        
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
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt     
    
    plotStyles = ['k-', 'kx-', 'k+-', 'k.-', 'k*-']
    
    N, resultsDir, outputDir, recordStep, startDate, endDate, prefix, targetGraph, breakSize, numEpsilons, M, matchAlpha, matchAlg, numInds = loadParams(0) 
    inds = range(numInds)
    numRecordSteps = int((endDate-startDate)/recordStep)+1
    
    t = 0
    for i in range(numEpsilons): 
        thetaArray, distArray = loadThetaArray(N, resultsDir, i)
        if thetaArray.shape[0] == N: 
            t = i   
    
    #We store: number of detections, CT detections, rand detections, infectives, max componnent size, num components, edges
    numMeasures = 7 
    thetas = []
    measures = numpy.zeros((len(inds), numMeasures, N, numRecordSteps))
    objectives = numpy.zeros((len(inds), numRecordSteps, N)) 
    idealMeasures = numpy.zeros((len(inds), numMeasures, numRecordSteps))

    plotInd = 0 
    timeInds = [3, 6, 10]
    
    for ind in inds: 
        logging.debug("ind=" + str(ind))
        N, resultsDir, outputDir, recordStep, startDate, endDate, prefix, targetGraph, breakSize, numEpsilons, M, matchAlpha, matchAlg, numInds = loadParams(ind)
        
        times = numpy.arange(startDate, endDate+1, recordStep) 
        realTheta, sigmaTheta, purtTheta = HIVModelUtils.toyTheta()
        thetaArray, distArray = loadThetaArray(N, resultsDir, t)
        thetas.append(thetaArray)
        print(thetaArray) 
        
        resultsFileName = outputDir + "IdealStats.pkl"
        stats = Util.loadPickle(resultsFileName)  
        vertexArrayIdeal, idealInfectedIndices, idealRemovedIndices, idealContactGraphStats, idealRemovedGraphStats = stats 
       
        graphStats = GraphStatistics()
        idealMeasures[ind, 0, :] = vertexArrayIdeal[:, 0]
        idealMeasures[ind, 1, :] = vertexArrayIdeal[:, 5]
        idealMeasures[ind, 2, :] = vertexArrayIdeal[:, 6]
        idealMeasures[ind, 4, :] = idealRemovedGraphStats[:, graphStats.numComponentsIndex]
        idealMeasures[ind, 5, :] = idealRemovedGraphStats[:, graphStats.maxComponentSizeIndex]
        idealMeasures[ind, 6, :] = idealRemovedGraphStats[:, graphStats.numEdgesIndex]
          
        
        for i in range(thetaArray.shape[0]): 
            resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
            stats = Util.loadPickle(resultsFileName)
            
            times, vertexArray, infectedIndices, removedGraphStats, objs, graphDists, labelDists = stats 
    
            measures[ind, 0, i, :] = vertexArray[:, 0]
            measures[ind, 1, i, :] = vertexArray[:, 5]
            measures[ind, 2, i, :] = vertexArray[:, 6]
            measures[ind, 3, i, :] = numpy.array([len(x) for x in infectedIndices])
            measures[ind, 4, i, :] = removedGraphStats[:, graphStats.numComponentsIndex]
            measures[ind, 5, i, :] = removedGraphStats[:, graphStats.maxComponentSizeIndex]
            measures[ind, 6, i, :] = removedGraphStats[:, graphStats.numEdgesIndex]
            
            #objectives[inds, i, :] = objs 

    
        times = times - numpy.min(times)
        logging.debug("times="+str(times))
        
        meanMeasures = numpy.mean(measures, 2)
        stdMeasures = numpy.std(measures, 2)

        #Infections and detections 
        plt.figure(plotInd)    
        if not processReal: 
            numInfects = [len(x) for x in idealInfectedIndices]
            plt.errorbar(times, meanMeasures[ind, 3, :], yerr=stdMeasures[ind, 3, :], label="est. infectives") 
            plt.plot(times, numInfects, "r", label="infectives")
        
        plt.errorbar(times, meanMeasures[ind, 0, :], yerr=stdMeasures[ind, 0, :], label="est. detections") 
        plt.xlabel("time (days)")
        plt.plot(times, idealMeasures[ind, 0, :], "k", label="detections")
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
        
        #Contact tracing rand random detections 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, 1, :], yerr=stdMeasures[ind, 1, :], label="est. CT detections") 
        plt.plot(times, idealMeasures[ind, 1, :], "r", label="CT detections")
    
        plt.errorbar(times, meanMeasures[ind, 2, :], yerr=stdMeasures[ind, 2, :], label="est. rand detections") 
        plt.xlabel("time (days)")
        plt.ylabel("detections")
        plt.plot(times, idealMeasures[ind, 2, :], "k", label="rand detections")
        if not processReal: 
            lims = plt.xlim()
            plt.xlim([0, lims[1]]) 
        plt.legend(loc="upper left")
        plt.savefig(outputDir + prefix + "CTRandDetects.eps")
        plotInd += 1
        
        #Number of components 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, 4, :], yerr=stdMeasures[ind, 5, :]) 
        plt.xlabel("time (days)")
        plt.ylabel("num components")
        plt.plot(times, idealMeasures[ind, 4, :], "r")
        plotInd += 1
        
        #Max component size 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, 5, :], yerr=stdMeasures[ind, 4, :]) 
        plt.xlabel("time (days)")
        plt.ylabel("max component size")
        plt.plot(times, idealMeasures[ind, 5, :], "r")
        plotInd += 1
        
        #Num edges 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, 6, :], yerr=stdMeasures[ind, 6, :]) 
        plt.xlabel("time (days)")
        plt.ylabel("number of edges")
        plt.plot(times, idealMeasures[ind, 6, :], "r")
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
        
        print(meanMeasures[ind, :, timeInds])
    
    #Print the table of thetas 
    thetas = numpy.array(thetas)
    meanThetas = numpy.mean(thetas, 1)
    stdThetas = numpy.std(thetas, 1)
    table = Latex.array2DToRows(meanThetas.T, stdThetas.T, precision=3)
    rowNames = ["$|\\mathcal{I}_0 |$", "$\\alpha$", "$\\gamma$", "$\\beta$", "$\\lambda$",  "$\\sigma$"]
    table = Latex.addRowNames(rowNames, table)
    print(table)    
    
    #Now print the graph properties 
    idealTable = []
    tableMeanArray = [] 
    tableStdArray = [] 
    for ind in inds: 
        idealTable.append(idealMeasures[ind, :, timeInds])
        tableMeanArray.append(meanMeasures[ind, :, timeInds])
        tableStdArray.append(stdMeasures[ind, :, timeInds])
        
    idealTable = numpy.vstack(idealTable).T
    tableMeanArray = numpy.vstack(tableMeanArray).T
    tableStdArray = numpy.vstack(tableStdArray).T
    
    rowNames = ["$|\\mathcal{R}_T |$.", "CT", "Rand", "$|\\mathcal{I}_0 |$", "MC size", "Num comp.", "Edges"]
    idealTable = Latex.array2DToRows(idealTable, precision=1)
    idealTable = Latex.addRowNames(rowNames, idealTable)
    print(idealTable)  
    
    rowNames = [x + " est." for x in rowNames]
    table = Latex.array2DToRows(tableMeanArray, tableStdArray, precision=1)
    table = Latex.addRowNames(rowNames, table)
    print(table)
    

    
    plt.show()
