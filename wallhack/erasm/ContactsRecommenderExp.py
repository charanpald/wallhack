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

#go through: different values of sigma, different algs, do model selection 

k = 8
maxItems = 10
overwrite = True
datasets = ["Keyword", "Doc"]
learners = [("SoftImpute", IterativeSoftImpute(k=k)), ("WRMF", WeightedMf(k=k))]
#learners = [("WRMF", WeightedMf(k=k))]
resultsDir = PathDefaults.getOutputDir() + "coauthors/"
contactsFilename = PathDefaults.getDataDir() + "reference/contacts_anonymised.tsv"
interestsFilename = PathDefaults.getDataDir() + "reference/author_interest"

for dataset in datasets: 
    for learnerName, learner in learners: 
        
        outputFilename = resultsDir + "Results_" + learnerName + "_" + dataset + ".npz"        
        fileLock = FileLock(outputFilename)  
            
        if not (fileLock.isLocked() or fileLock.fileExists()) or overwrite: 
            fileLock.lock()        
        
            try: 
                X = DatasetUtils.mendeley(dataset=dataset)
                
                #Do some recommendation 
                if type(learner) == IterativeSoftImpute:  
                    X = X.toScipyCsc()
                    trainIterator = iter([X])
                    ZList = learner.learnModel(trainIterator)    
                    U, s, V = ZList.next()
                    U = U*s
                else: 
                    X = X.toScipyCsr()
                    learner.learnModel(X)
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
                
                similaritiesFileName = resultsDir + "Recommendations.csv"
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
                minScore = 0.1
                minContacts = 3
                minAcceptableSims = 3
                
                similaritiesFileName = resultsDir + "Recommendations.csv"
                
                contacts = read_contacts(contactsFilename)
                research_interests = read_interests(interestsFilename)
                sims = read_similar_authors(similaritiesFileName, minScore)
                
                logging.debug('Evaluating against contacts...')
                precisions, recalls = evaluate_against_contacts(sims, contacts, minContacts)
                
                #logging.debug('Evaluating against research interests...') 
                #precisions = evaluate_against_research_interests(sims, research_interests, minAcceptableSims)
                
                logging.debug("Precisions: " + precisions)
                logging.debug("Recalls: " + recalls)
                
                numpy.savez(outputFilename, precisions, recalls)
                logging.debug("Saved precisions/recalls on contacts as " + outputFilename)
        
            finally: 
                fileLock.unlock()
        else: 
            logging.debug("File is locked or already computed: " + outputFilename)      
