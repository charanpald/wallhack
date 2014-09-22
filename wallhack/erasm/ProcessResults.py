
import numpy 
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.Latex import Latex 


"""
Just print the results for the contacts recommender. 
"""
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

sigmas1 = [0.1, 0.15, 0.2]
sigmas2 =  [0.7, 0.8, 0.9]
datasets = ["Keyword", "Doc"]
learnerNames = ["SoftImpute", "WRMF"]

resultsDir = PathDefaults.getOutputDir() + "coauthors/"


rowNames = [] 
contactResults = []
interestResults = []

for dataset in datasets: 
    
    if dataset == "Doc": 
        sigmas = sigmas1 
    else: 
        sigmas = sigmas2
    for learnerName in learnerNames: 
        for sigma in sigmas:         
        
            
            outputFilename = resultsDir + "Results_" + learnerName + "_" + dataset + "_sigma=" + str(sigma) + ".npz"  
            
            try : 
                data = numpy.load(outputFilename)
                
                meanStatsContacts, meanStatsInterests = data["arr_0"], data["arr_1"]
                
                print(outputFilename)
                #print(meanStatsContacts)
                print(meanStatsInterests)
                
                rowNames.append(learnerName + " " + dataset + " $\sigma=" + str(sigma) + "$")
                contactResults.append(meanStatsContacts[[2, 3, 4, 5, 7, 8, 9, 10, 12]])
                interestResults.append(meanStatsInterests[[1, 2, 3, 4, 6]])
            except: 
                print("File not found: " + outputFilename)                


contactResults = numpy.array(contactResults)
interestResults = numpy.array(interestResults)


contactsColNames = ["p@1", "p@3", "p@5", "p@10", "r@1", "r@3", "r@5", "r@10", "f1"]
contactsTable = Latex.addRowNames(rowNames, Latex.array2DToRows(contactResults, precision=4))
print(Latex.listToRow(contactsColNames))
print(contactsTable)


interestsColNames = ["p@1", "p@3", "p@5", "p@10", "j@10"]
interestsTable = Latex.addRowNames(rowNames, Latex.array2DToRows(interestResults))
print(Latex.listToRow(interestsColNames))
print(interestsTable)