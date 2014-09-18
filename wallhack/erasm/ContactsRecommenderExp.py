import numpy 
import logging 
import pickle 
import csv
import sys 
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from sandbox.recommendation.WeightedMf import WeightedMf
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.FileLock import FileLock
from wallhack.rankingexp.DatasetUtils import DatasetUtils 
from wallhack.erasm.Evaluator import evaluate_against_contacts, evaluate_against_research_interests, read_contacts, read_interests, read_similar_authors

"""
Use Mendeley author-documents and author-keywords to recommend contacts. 

"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#Do model selection? 

k = 128
maxItems = 10
minScore = 0.1
minContacts = 3
minAcceptableSims = 3
maxIterations = 50 
alpha = 0.2


sigmas1 = [0.1, 0.15, 0.2]
sigmas2 =  [0.7, 0.8, 0.9]

softImpute = IterativeSoftImpute(k=k, postProcess=True)
wrmf = WeightedMf(k=k, maxIterations=maxIterations, alpha=alpha)

overwrite = False
datasets = ["Keyword", "Doc"]
learners = [("SoftImpute", softImpute), ("WRMF", wrmf)]
#learners = [("WRMF", WeightedMf(k=k))]
resultsDir = PathDefaults.getOutputDir() + "coauthors/"
contactsFilename = PathDefaults.getDataDir() + "reference/contacts_anonymised.tsv"
interestsFilename = PathDefaults.getDataDir() + "reference/author_interest"

for dataset in datasets: 
    
    if dataset == "Doc": 
        sigmas = sigmas1 
    else: 
        sigmas = sigmas2
    
    for sigma in sigmas: 
        X = DatasetUtils.mendeley(dataset=dataset, sigma=sigma)
        
        for learnerName, learner in learners: 
            
            outputFilename = resultsDir + "Results_" + learnerName + "_" + dataset + "_sigma=" + str(sigma) + ".npz"        
            fileLock = FileLock(outputFilename)  
                
            if not (fileLock.isLocked() or fileLock.fileExists()) or overwrite: 
                fileLock.lock()       
                
                logging.debug(learner)
            
                try: 
                                    
                    #Do some recommendation 
                    if type(learner) == IterativeSoftImpute:  
                        trainX = X.toScipyCsc()
                        trainIterator = iter([trainX])
                        ZList = learner.learnModel(trainIterator)    
                        U, s, V = ZList.next()
                        U = U*s
                    else: 
                        trainX = X.toScipyCsr()
                        learner.learnModel(trainX)
                        U = learner.U 
                        V = learner.V 
                
                    U = numpy.ascontiguousarray(U)
                    V = numpy.ascontiguousarray(V)
                    
                    orderedItems, scores = MCEvaluator.recommendAtk(U, V, maxItems, verbose=True)
                    
                    #Now let's write out the similarities file 
                    logging.debug("Generating recommendations for authors")
                    authorIndexerFilename = PathDefaults.getDataDir() + "reference/authorIndexer" + dataset + ".pkl"
                    authorIndexerFile = open(authorIndexerFilename)
                    authorIndexer = pickle.load(authorIndexerFile)
                    authorIndexerFile.close()
                    logging.debug("Loaded author indexer")
                    
                    reverseIndexer = authorIndexer.reverseTranslateDict()
                    
                    similaritiesFileName = resultsDir + "Recommendations_" + learnerName + "_" + dataset + "_sigma=" + str(sigma) + ".csv" 
                    outputFile = open(similaritiesFileName, "w")
                    csvFile = csv.writer(outputFile, delimiter='\t')
                    
                    for i in range(orderedItems.shape[0]):
                        if i % 10000 == 0 : 
                            logging.debug("Iteration: " + str(i))
                            
                        row = [reverseIndexer[i]]                
                        
                        #Check author isn't recommended him/herself
                        for j in range(orderedItems.shape[1]): 
                            row = [reverseIndexer[i], reverseIndexer[orderedItems[i, j]], scores[i, j]]
                        
                            csvFile.writerow(row)
                            
                    outputFile.close()
                    logging.debug("Wrote recommendations to " + similaritiesFileName)
                    
                    #Figure out how good the recommendations are on the contacts network  
                    similaritiesFileName = resultsDir + "Recommendations.csv"
                    
                    contacts = read_contacts(contactsFilename)
                    research_interests = read_interests(interestsFilename)
                    sims = read_similar_authors(similaritiesFileName, minScore)
                    
                    logging.debug('Evaluating against contacts...')
                    precisions, recalls, f1 = evaluate_against_contacts(sims, contacts, minContacts)
                    
                    logging.debug('Evaluating against research interests...') 
                    precisions_interests, jaccard_10 = evaluate_against_research_interests(sims, research_interests, minAcceptableSims)
                    
                    logging.debug("Precisions: " + str(precisions))
                    logging.debug("Recalls: " + str(recalls))
                    logging.debug("F1: " + str(f1))
                    
                    numpy.savez(outputFilename, precisions, recalls, numpy.array([f1]), precisions_interests, jaccard_10)
                    logging.debug("Saved precisions/recalls on contacts/interests as " + outputFilename)
            
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + outputFilename)      
