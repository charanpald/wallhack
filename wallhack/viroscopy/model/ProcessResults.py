import numpy
import logging
import sys 
import multiprocessing 
import os
from apgl.graph.GraphStatistics import GraphStatistics 
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Util import Util 
from sandbox.util.Latex import Latex 
from sandbox.util.FileLock import FileLock
from sandbox.predictors.ABCSMC import loadThetaArray 
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils

assert False, "Must run with -O flag"
FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

processReal = False 
saveResults = False 

def loadParams(ind): 
    if processReal: 
        resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/theta" + str(ind) + "/"
        outputDir = resultsDir + "stats/"
        
        N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale = HIVModelUtils.realABCParams(True)
        startDate, endDate, recordStep, M, targetGraph, numInds = HIVModelUtils.realSimulationParams(test=True, ind=ind)
        realTheta, sigmaTheta, pertTheta = HIVModelUtils.estimatedRealTheta(ind)
        numInds=2
        prefix = "Real"
    else: 
        resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/theta/"
        outputDir = resultsDir + "stats/"        
        
        N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale = HIVModelUtils.toyABCParams()
        startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams(test=True)
        realTheta, sigmaTheta, pertTheta = HIVModelUtils.toyTheta()
        prefix = "Toy"
        numInds = 1

    breakSize = (targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size - targetGraph.subgraph(targetGraph.removedIndsAt(startDate)).size)  * breakScale       
        
    return N, resultsDir, outputDir, recordStep, startDate, endDate, prefix, targetGraph, breakSize, numEpsilons, M, matchAlpha, matchAlg, numInds

def saveStats(args):    
    i, theta = args 
    
    resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
    lock = FileLock(resultsFileName)
    
    if not lock.fileExists() and not lock.isLocked():    
        lock.lock()
         
        model = HIVModelUtils.createModel(targetGraph, startDate, endDate, recordStep, M, matchAlpha, breakSize, matchAlg, theta=thetaArray[i])
        times, infectedIndices, removedIndices, graph, compTimes, graphMetrics = HIVModelUtils.simulate(model)
        times = numpy.arange(startDate, endDate+1, recordStep)
        vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats, finalRemovedDegrees = HIVModelUtils.generateStatistics(graph, times)
        stats = times, vertexArray, infectedIndices, removedGraphStats, finalRemovedDegrees, graphMetrics.objectives, compTimes
        
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
            thetaArray, objArray = loadThetaArray(N, resultsDir, i)
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
        vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats, finalRemovedDegrees = HIVModelUtils.generateStatistics(targetGraph, times)
        stats = vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats, finalRemovedDegrees
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
    
    #We store: number of detections, CT detections, rand detections, infectives, max componnent size, num components, edges, objectives
    numMeasures = 12
    numTimings = 2
    thetas = []
    measures = numpy.zeros((len(inds), numMeasures, N, numRecordSteps))
    idealMeasures = numpy.zeros((len(inds), numMeasures, numRecordSteps))
    timings = numpy.zeros((len(inds), numTimings, N)) 
    
    numDegrees = 20
    degreeDists = numpy.zeros((len(inds), numDegrees, N))
    idealDegreeDists = numpy.zeros((len(inds), numDegrees))
    
    #Note all the inds 
    numDetectsInd = 0 
    maleInd = 1 
    femaleInd = 2 
    heteroInd = 3 
    biInd = 4 
    randDetectInd = 5 
    contactDetectInd = 6 
    infectedInd = 7
    numCompsInd = 8 
    maxCompSizeInd = 9 
    numEdgesInd = 10
    objsInd = 11
    
    
    plotInd = 0 
    if processReal: 
        timeInds = [5, 6]
    else: 
        timeInds = [10, 11, 12, 13]    
    
    for ind in inds: 
        logging.debug("ind=" + str(ind))
        
        N, resultsDir, outputDir, recordStep, startDate, endDate, prefix, targetGraph, breakSize, numEpsilons, M, matchAlpha, matchAlg, numInds = loadParams(ind)
        
        #Find the max number t for which we have a complete set of particles 
        t = 0
        for i in range(numEpsilons): 
            thetaArray, objArray = loadThetaArray(N, resultsDir, i)
            if thetaArray.shape[0] == N: 
                t = i
        logging.debug("Using particle number " + str(t))        
        
        times = numpy.arange(startDate, endDate+1, recordStep) 
        realTheta, sigmaTheta, purtTheta = HIVModelUtils.toyTheta()
        thetaArray, objArray = loadThetaArray(N, resultsDir, t)
        thetas.append(thetaArray)
        print(thetaArray) 
        
        resultsFileName = outputDir + "IdealStats.pkl"
        stats = Util.loadPickle(resultsFileName)  
        vertexArrayIdeal, idealInfectedIndices, idealRemovedIndices, idealContactGraphStats, idealRemovedGraphStats, idealFinalRemovedDegrees = stats 
       
        graphStats = GraphStatistics()
        idealMeasures[ind, numDetectsInd, :] = vertexArrayIdeal[:, numDetectsInd]
        idealMeasures[ind, maleInd, :] = vertexArrayIdeal[:, maleInd]
        idealMeasures[ind, femaleInd, :] = vertexArrayIdeal[:, femaleInd]
        idealMeasures[ind, heteroInd, :] = vertexArrayIdeal[:, heteroInd]
        idealMeasures[ind, biInd, :] = vertexArrayIdeal[:, biInd]
        idealMeasures[ind, randDetectInd, :] = vertexArrayIdeal[:, randDetectInd]
        idealMeasures[ind, contactDetectInd, :] = vertexArrayIdeal[:, contactDetectInd]
        idealMeasures[ind, numCompsInd, :] = idealRemovedGraphStats[:, graphStats.numComponentsIndex]
        idealMeasures[ind, maxCompSizeInd, :] = idealRemovedGraphStats[:, graphStats.maxComponentSizeIndex]
        idealMeasures[ind, numEdgesInd, :] = idealRemovedGraphStats[:, graphStats.numEdgesIndex]
        
        maxDegrees = min(idealFinalRemovedDegrees.shape[0], numDegrees)
        idealDegreeDists[ind, 0:maxDegrees] = idealFinalRemovedDegrees[0:maxDegrees]
        
        for i in range(thetaArray.shape[0]): 
            resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
            stats = Util.loadPickle(resultsFileName)
            
            times, vertexArray, infectedIndices, removedGraphStats, finalRemovedDegrees, objs, compTimes = stats 
    
            measures[ind, numDetectsInd, i, :] = vertexArray[:, numDetectsInd]
            measures[ind, maleInd, i, :] = vertexArray[:, maleInd]
            measures[ind, femaleInd, i, :] = vertexArray[:, femaleInd]
            measures[ind, heteroInd, i, :] = vertexArray[:, heteroInd]
            measures[ind, biInd, i, :] = vertexArray[:, biInd]
            measures[ind, randDetectInd, i, :] = vertexArray[:, randDetectInd]
            measures[ind, contactDetectInd, i, :] = vertexArray[:, contactDetectInd]
            measures[ind, infectedInd, i, :] = numpy.array([len(x) for x in infectedIndices])
            measures[ind, numCompsInd, i, :] = removedGraphStats[:, graphStats.numComponentsIndex]
            measures[ind, maxCompSizeInd, i, :] = removedGraphStats[:, graphStats.maxComponentSizeIndex]
            measures[ind, numEdgesInd, i, :] = removedGraphStats[:, graphStats.numEdgesIndex]
            measures[ind, objsInd, i, 1:] = objs
            
            maxDegrees = min(finalRemovedDegrees.shape[0], numDegrees)
            degreeDists[ind, 0:maxDegrees, i] = finalRemovedDegrees[0:maxDegrees]
            
            #objectives[inds, i, :] = objs 
            timings[ind, :, i] = compTimes

    
        times = times - numpy.min(times)
        logging.debug("times="+str(times))
        
        logging.debug("computational times="+str(numpy.mean(timings, 2)))
        
        meanMeasures = numpy.mean(measures, 2)
        stdMeasures = numpy.std(measures, 2)

        #Infections and detections 
        plt.figure(plotInd)    
        if not processReal: 
            numInfects = [len(x) for x in idealInfectedIndices]
            plt.errorbar(times, meanMeasures[ind, infectedInd, :], color="k", yerr=stdMeasures[ind, infectedInd, :], label="est. infectives") 
            plt.plot(times, numInfects, "k--", label="infectives")
        
        plt.errorbar(times, meanMeasures[ind, numDetectsInd, :], color="r", yerr=stdMeasures[ind, numDetectsInd, :], label="est. detections") 
        plt.plot(times, idealMeasures[ind, numDetectsInd, :], "r--", label="detections")
        plt.xlabel("time (days)")
        
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
      
        #Gender 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, maleInd, :], yerr=stdMeasures[ind, maleInd, :], label="est. male") 
        plt.plot(times, idealMeasures[ind, maleInd, :], "r", label="male")
    
        plt.errorbar(times, meanMeasures[ind, femaleInd, :], yerr=stdMeasures[ind, femaleInd, :], label="est. female") 
        plt.plot(times, idealMeasures[ind, femaleInd, :], "k", label="female")
    
        plt.xlabel("time (days)")
        plt.ylabel("detections")
        plt.legend(loc="upper left")
        plotInd += 1        
      
        #Orientation
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, heteroInd, :], yerr=stdMeasures[ind, heteroInd, :], label="est. hetero") 
        plt.plot(times, idealMeasures[ind, heteroInd, :], "r", label="hetero")
    
        plt.errorbar(times, meanMeasures[ind, biInd, :], yerr=stdMeasures[ind, biInd, :], label="est. bi") 
        plt.plot(times, idealMeasures[ind, biInd, :], "k", label="bi")
    
        plt.xlabel("time (days)")
        plt.ylabel("detections")
        plt.legend(loc="upper left")
        plotInd += 1      
      
        #Contact tracing rand random detections            
        plt.figure(plotInd)
        if processReal: 
            plt.errorbar(times, meanMeasures[ind, numDetectsInd, :], color="k", yerr=stdMeasures[ind, numDetectsInd, :],  label="est. detections") 
            plt.plot(times, idealMeasures[ind, numDetectsInd, :], "k--", label="detections")        
        
        plt.errorbar(times, meanMeasures[ind, contactDetectInd, :], color="r", yerr=stdMeasures[ind, contactDetectInd, :], label="est. CT detections") 
        plt.plot(times, idealMeasures[ind, contactDetectInd, :], "r--", label="CT detections")
    
        plt.errorbar(times, meanMeasures[ind, randDetectInd, :], color="b", yerr=stdMeasures[ind, randDetectInd, :], label="est. random detections") 
        plt.plot(times, idealMeasures[ind, randDetectInd, :], "b--", label="random detections")
        plt.xlabel("time (days)")
        plt.ylabel("detections")
        
        if not processReal: 
            lims = plt.xlim()
            plt.xlim([0, lims[1]]) 
        plt.legend(loc="upper left")
        plt.savefig(outputDir + prefix + "CTRandDetects" + str(ind) +  ".eps")
        plotInd += 1
        
        #Number of components 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, numCompsInd, :], yerr=stdMeasures[ind, numCompsInd, :]) 
        plt.xlabel("time (days)")
        plt.ylabel("num components")
        plt.plot(times, idealMeasures[ind, numCompsInd, :], "r")
        plotInd += 1
        
        #Max component size 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, maxCompSizeInd, :], yerr=stdMeasures[ind, maxCompSizeInd, :]) 
        plt.xlabel("time (days)")
        plt.ylabel("max component size")
        plt.plot(times, idealMeasures[ind, maxCompSizeInd, :], "r")
        plotInd += 1
        
        #Num edges 
        plt.figure(plotInd)
        plt.errorbar(times, meanMeasures[ind, numEdgesInd, :], yerr=stdMeasures[ind, numEdgesInd, :]) 
        plt.xlabel("time (days)")
        plt.ylabel("number of edges")
        plt.plot(times, idealMeasures[ind, numEdgesInd, :], "r")
        plotInd += 1
        
        #Objectives 
        plt.figure(plotInd)
        plt.errorbar(times[1:], meanMeasures[ind, objsInd, 1:], yerr=stdMeasures[ind, objsInd, 1:]) 
        plt.xlabel("time (days)")
        plt.ylabel("objectives")
        plotInd += 1
        
        #Degrees 
        meanDegreeDists = numpy.mean(degreeDists, 2)
        stdDegreeDists = numpy.std(degreeDists, 2)        
        
        plt.figure(plotInd)
        plt.errorbar(numpy.arange(numDegrees), meanDegreeDists[ind, :], yerr=stdDegreeDists[ind, :], color="k") 
        plt.plot(numpy.arange(numDegrees), idealDegreeDists[ind,  :], "k--")
        plt.xlabel("degree")
        plt.ylabel("frequency")
        plotInd += 1
        

    #Print the table of thetas 
    thetas = numpy.array(thetas)
    meanThetas = numpy.mean(thetas, 1)
    stdThetas = numpy.std(thetas, 1)
    table = Latex.array2DToRows(meanThetas.T, stdThetas.T, precision=4)
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
      
    rowNames = ["$|\\mathcal{R}_{t_0}|$.", "male", "female", "hetero", "bi", "RD", "CT", "$|\\mathcal{I}_{t_0}|$", "NC", "LC", "$|E|$", "objs"]
    idealTable = Latex.array2DToRows(idealTable, precision=0)
    idealTable = Latex.addRowNames(rowNames, idealTable)
    print(idealTable)  
    
    rowNames = [x + " est." for x in rowNames]
    table = Latex.array2DToRows(tableMeanArray, tableStdArray, precision=0)
    table = Latex.addRowNames(rowNames, table)
    print(table)
    
    #Now print timings 
    rowNames = [str(x) for x in range(numInds)]
    table = Latex.array2DToRows(numpy.mean(timings, 2), numpy.std(timings, 2), precision=1)
    table = Latex.addRowNames(rowNames, table)
    print(table)    
    
    plt.show()
