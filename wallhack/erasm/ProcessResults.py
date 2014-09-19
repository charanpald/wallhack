
import numpy 
from sandbox.util.PathDefaults import PathDefaults 

"""
Just print the results for the contacts recommender. 
"""
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

sigmas1 = [0.1, 0.15, 0.2]
sigmas2 =  [0.7, 0.8, 0.9]
datasets = ["Keyword", "Doc"]
learnerNames = ["SoftImpute", "WRMF"]

resultsDir = PathDefaults.getOutputDir() + "coauthors/"

for dataset in datasets: 
    
    if dataset == "Doc": 
        sigmas = sigmas1 
    else: 
        sigmas = sigmas2
    
    for sigma in sigmas:         
        for learnerName in learnerNames: 
            
            outputFilename = resultsDir + "Results_" + learnerName + "_" + dataset + "_sigma=" + str(sigma) + ".npz"  
            
            try : 
                data = numpy.load(outputFilename)
                
                meanStatsContacts, meanStatsInterests = data["arr_0"], data["arr_1"]
                
                print(outputFilename)
                print(meanStatsContacts)
                print(meanStatsInterests)
            except: 
                print("File not found: " + outputFilename)                
                