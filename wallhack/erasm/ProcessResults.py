
import numpy 
from sandbox.util.PathDefaults import PathDefaults 

"""
Just print the results for the contacts recommender. 
"""
sigmas1 = [0.05, 0.1, 0.2]
sigmas2 = [0.5, 0.8]
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
                
                precisions, recalls, f1, precisionsInterests, jaccardInterests = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"]
                
                print(outputFilename)
                print(precisions, recalls, f1, precisionsInterests, jaccardInterests)
            except: 
                print("File not found: " + outputFilename)                
                