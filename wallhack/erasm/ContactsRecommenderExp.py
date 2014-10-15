import numpy 
import logging 
import pickle 
import csv
import sys 
import os
import argparse 
import multiprocessing
from mrec.item_similarity.knn import CosineKNNRecommender
from mrec.item_similarity.slim import SLIM
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
To run the parallel version of MLAUC you have to increase the amount of shared memory using 
sudo sysctl -w kernel.shmmax=2147483648
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def saveResults(orderedItems, scores, dataset, similaritiesFileName, contactsFilename, interestsFilename, minScore, minContacts, minAcceptableSims): 
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
            if orderedItems[i, j] != i:
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
    
    return meanStatsContacts, meanStatsInterests 


parser = argparse.ArgumentParser(description='Recommend coauthors')
parser.add_argument("--alg", type=str, help="Choice of algorithm (default: %(default)s)", default="SoftImpute")
parser.add_argument("--k", type=int, help="Number of latent factors (default: %(default)s)", default=128)
parser.add_argument("--modelSelect", action="store_true", help="Perform model selection (default: %(default)s)", default=False)
parser.add_argument("--overwrite", action="store_true", help="Overwrite results (default: %(default)s)", default=False)
parser.add_argument("--processes", type=int, help="Number of processes (default: %(default)s)", default=multiprocessing.cpu_count())

args = parser.parse_args()

k = args.k
maxItems = 10
minScore = 0.0
minContacts = 3
minAcceptableSims = 3
maxIterations = 30 
alpha = 0.2
numProcesses = 2
modelSelectSamples = 10**6

modelSelect = args.modelSelect
folds = 3
ks = numpy.array([64, 128, 256])
rhosSi = numpy.linspace(1.0, 0.0, 5)

overwrite = args.overwrite
datasets = ["Keyword", "Document"]

resultsDir = PathDefaults.getOutputDir() + "coauthors/"
contactsFilename = PathDefaults.getDataDir() + "reference/contacts_anonymised.tsv"
interestsFilename = PathDefaults.getDataDir() + "reference/author_interest"


#Create all the recommendation algorithms
softImpute = IterativeSoftImpute(k=k, postProcess=True, svdAlg="rsvd")
softImpute.maxIterations = maxIterations
softImpute.metric = "f1" 
softImpute.q = 3
softImpute.p = 10
softImpute.rho = 0.1
softImpute.eps = 10**-4 
softImpute.numProcesses = args.processes

wrmf = WeightedMf(k=k, maxIterations=maxIterations, alpha=1.0)
wrmf.ks = ks
wrmf.folds = folds 
wrmf.lmbdas = 2.0**-numpy.arange(-1, 12, 2)
wrmf.metric = "f1" 
wrmf.numProcesses = args.processes

maxLocalAuc = MaxLocalAUC(k=k, w=0.9, maxIterations=50, lmbdaU=0.1, lmbdaV=0.1, stochastic=True)
maxLocalAuc.numRowSamples = 10
maxLocalAuc.parallelSGD = True
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.ks = ks
maxLocalAuc.folds = folds
maxLocalAuc.metric = "f1"
maxLocalAuc.numProcesses = args.processes

kNeighbours = 25
knn = CosineKNNRecommender(kNeighbours)

numFeatures = 200
slim = SLIM(num_selected_features=numFeatures)

learners = [("SoftImpute", softImpute), ("WRMF", wrmf), ("KNN", knn), ("MLAUC", maxLocalAuc), ("SLIM", slim)]

#Figure out the correct learner 
for tempLearnerName, tempLearner in learners: 
    if args.alg == tempLearnerName: 
        learnerName = tempLearnerName
        learner = tempLearner 

if "learner" not in globals(): 
    raise ValueError("Learner not found: " + learnerName)

os.system('taskset -p 0xffffffff %d' % os.getpid())

for dataset in datasets: 
    X = DatasetUtils.mendeley2(minNnzRows=0, dataset=dataset)

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
            elif type(learner) == CosineKNNRecommender or type(learner) == SLIM: 
                fastTrainX = fast_sparse_matrix(X.toScipyCsr())
                trainX = X.toScipyCsr()
                m, n = trainX.shape
                learner.fit(fastTrainX)
                
                recommendations = learner.range_recommend_items(trainX, 0, m, max_items=maxItems)
                
                orderedItems = numpy.zeros((m, maxItems), numpy.int)
                scores = numpy.zeros((m, maxItems))
                
                for i in range(m):
                    itemScores = numpy.array(recommendations[i])
                    if itemScores.shape[0] != 0: 
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
                orderedItems2, scores2 = MCEvaluator.recommendAtk(U.dot(V.T.dot(V)), U, maxItems, verbose=True)
            else: 
                orderedItems2 = orderedItems 
                scores2 = scores 
            
            #Normalise scores 
            scores /= numpy.max(scores)                
            
            meanStatsContacts, meanStatsInterests = saveResults(orderedItems, scores, dataset, similaritiesFileName, contactsFilename, interestsFilename, minScore, minContacts, minAcceptableSims)
            meanStatsContacts2, meanStatsInterests2 = saveResults(orderedItems2, scores2, dataset, similaritiesFileName, contactsFilename, interestsFilename, minScore, minContacts, minAcceptableSims)

            numpy.savez(outputFilename, meanStatsContacts, meanStatsInterests, meanStatsContacts2, meanStatsInterests2)
            logging.debug("Saved precisions/recalls on contacts/interests as " + outputFilename)
    
        finally: 
            fileLock.unlock()
    else: 
        logging.debug("File is locked or already computed: " + outputFilename)      
