import numpy 
import logging 
import pickle 
import csv
import sys 
import os
from mrec.item_similarity.knn import CosineKNNRecommender
from mrec.sparse import fast_sparse_matrix
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from sandbox.recommendation.WeightedMf import WeightedMf
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.FileLock import FileLock
from sandbox.util.Sampling import Sampling 
from wallhack.rankingexp.DatasetUtils import DatasetUtils 
from wallhack.erasm.Evaluator import evaluate_against_contacts, evaluate_against_research_interests, read_contacts, read_interests, read_similar_authors


"""
Use Mendeley author-documents and author-keywords to recommend contacts. 

"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#Do model selection? 

k = 64
maxItems = 10
minScore = 0.0
minContacts = 3
minAcceptableSims = 3
maxIterations = 30 
alpha = 0.2
numProcesses = 2
modelSelectSamples = 10**6

modelSelect = True
folds = 3
ks = numpy.array([k])
rhosSi = numpy.linspace(1.0, 0.0, 5)

softImpute = IterativeSoftImpute(k=k, postProcess=True, svdAlg="rsvd")
softImpute.maxIterations = maxIterations
softImpute.metric = "f1" 
softImpute.q = 3

wrmf = WeightedMf(k=k, maxIterations=maxIterations, alpha=1.0)
wrmf.ks = ks
wrmf.folds = folds 
wrmf.lmbdas = 2.0**-numpy.arange(-1, 12, 2)
wrmf.metric = "f1" 

maxLocalAuc = MaxLocalAUC(k=k, w=0.9, maxIterations=50, lmbdaU=0.1, lmbdaV=0.1, stochastic=True)
maxLocalAuc.numRowSamples = 10
maxLocalAuc.parallelSGD = True
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.ks = ks
maxLocalAuc.folds = folds
maxLocalAuc.metric = "f1"

kNeighbours = 25
knn = CosineKNNRecommender(kNeighbours)

"""
To run the parallel version of MLAUC you have to increase the amount of shared memory using 
sudo sysctl -w kernel.shmmax=2147483648
"""

overwrite = False
datasets = ["Keyword", "Document"]
learners = [("SoftImpute", softImpute), ("WRMF", wrmf), ("KNN", knn)]
#learners = [("MLAUC", maxLocalAuc)]
#learners = [("SoftImpute", softImpute)]
#learners = [("KNN", knn)]
resultsDir = PathDefaults.getOutputDir() + "coauthors/"
contactsFilename = PathDefaults.getDataDir() + "reference/contacts_anonymised.tsv"
interestsFilename = PathDefaults.getDataDir() + "reference/author_interest"

os.system('taskset -p 0xffffffff %d' % os.getpid())

for dataset in datasets: 
    X = DatasetUtils.mendeley2(minNnzRows=0, dataset=dataset)
    
    for learnerName, learner in learners: 
        outputFilename = resultsDir + "Results_" + learnerName + "_" + dataset + ".npz"  
        similaritiesFileName = resultsDir + "Recommendations_" + learnerName + "_" + dataset + ".csv" 
        fileLock = FileLock(outputFilename)  
            
        if not (fileLock.isLocked() or fileLock.fileExists()) or overwrite: 
            fileLock.lock()       
            
            logging.debug(learner)      
        
            try: 
                #Do some recommendation 
                if type(learner) == IterativeSoftImpute:  
                    trainX = X.toScipyCsc()
                    trainIterator = iter([trainX])
                             
                    if modelSelect: 
                        modelSelectX, userInds = Sampling.sampleUsers2(X, modelSelectSamples)
                        modelSelectX = modelSelectX.toScipyCsc()                            
                        cvInds = Sampling.randCrossValidation(folds, modelSelectX.nnz)
                        meanMetrics, stdMetrics = learner.modelSelect2(modelSelectX, rhosSi, ks, cvInds)
                    
                    ZList = learner.learnModel(trainIterator)    
                    U, s, V = ZList.next()
                    U = U*s
                elif type(learner) == WeightedMf:  
                    trainX = X.toScipyCsr()

                    if modelSelect:                     
                        modelSelectX, userInds = Sampling.sampleUsers2(X, modelSelectSamples)
                        modelSelectX = modelSelectX.toScipyCsc()  
                        meanMetrics, stdMetrics = learner.modelSelect(modelSelectX)                          
                    
                    learner.learnModel(trainX)
                    U = learner.U 
                    V = learner.V
                elif type(learner) == CosineKNNRecommender: 
                    #X, userInds = Sampling.sampleUsers2(X, 5000)
                    #print("Sampled X")
                    trainX = fast_sparse_matrix(X.toScipyCsr())
                    m, n = trainX.shape
                    learner.fit(trainX)
                    
                    recommendations = learner.range_recommend_items(trainX, 0, m, max_items=maxItems)
                    
                    orderedItems = numpy.zeros((m, maxItems), numpy.int)
                    scores = numpy.zeros((m, maxItems))
                    
                    for i in range(m):
                        itemScores = numpy.array(recommendations[i])
                        orderedItems[i, 0:itemScores.shape[0]] =  itemScores[:, 0]
                        scores[i, 0:itemScores.shape[0]] = itemScores[:, 1]
                        
                else: 
                    trainX = X
                    
                    if modelSelect: 
                        modelSelectX, userInds = Sampling.sampleUsers2(trainX, modelSelectSamples)
                        meanMetrics, stdMetrics = learner.modelSelect(modelSelectX)
                    
                    learner.learnModel(X)
                    U = learner.U 
                    V = learner.V 
                
                if type(learner) != CosineKNNRecommender:
                    U = numpy.ascontiguousarray(U)
                    V = numpy.ascontiguousarray(V)
                    
                    #Note that we compute UU^T for recommendations 
                    orderedItems, scores = MCEvaluator.recommendAtk(U, U, maxItems, verbose=True)
                
                #Now let's write out the similarities file 
                logging.debug("Generating recommendations for authors")
                authorIndexerFilename = PathDefaults.getDataDir() + "reference/authorIndexer" + dataset + ".pkl"
                authorIndexerFile = open(authorIndexerFilename)
                authorIndexer = pickle.load(authorIndexerFile)
                authorIndexerFile.close()
                logging.debug("Loaded author indexer")
                
                reverseIndexer = authorIndexer.reverseTranslateDict()
                
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
                contacts = read_contacts(contactsFilename)
                research_interests = read_interests(interestsFilename)
                sims = read_similar_authors(similaritiesFileName, minScore)
                
                logging.debug('Evaluating against contacts...')
                meanStatsContacts = evaluate_against_contacts(sims, contacts, minContacts)
                
                logging.debug('Evaluating against research interests...') 
                meanStatsInterests = evaluate_against_research_interests(sims, research_interests, minAcceptableSims)
                
                logging.debug("Mean stats on contacts: " + str(meanStatsContacts))
                logging.debug("Mean stats on interests:" + str(meanStatsInterests))
                
                numpy.savez(outputFilename, meanStatsContacts, meanStatsInterests)
                logging.debug("Saved precisions/recalls on contacts/interests as " + outputFilename)
        
            finally: 
                fileLock.unlock()
        else: 
            logging.debug("File is locked or already computed: " + outputFilename)      
