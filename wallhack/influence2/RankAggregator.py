import numpy 
import logging
import scipy.sparse 
import scipy.sparse.linalg
import array 
from sandbox.util.Util import Util 
from sandbox.util.Evaluator import Evaluator

class RankAggregator(object): 
    def __init__(self): 
        pass 
    
    @staticmethod 
    def spearmanFootrule(list1, list2): 
        """
        Compute the spearman footrule distance between two ranked lists. The lists 
        must be the same size. 
        """
        dist = 0 
        
        score1 = numpy.zeros(len(list1))
        score2 = numpy.zeros(len(list2))
        
        score1[list1] = numpy.arange(len(list1))        
        score2[list2] = numpy.arange(len(list2))        
        
        for i in range(len(list1)): 
            dist += abs(score1[i] - score2[i])
            
        dist /= (len(list1)**2)/2
        
        return dist 


    @staticmethod 
    def borda(list1, list2): 
        """
        Borda's method for combining rankings. 
        """
        score1 = numpy.zeros(len(list1))
        score2 = numpy.zeros(len(list2))
        
        score1[list1] = numpy.flipud(numpy.arange(len(list1)))     
        score2[list2] = numpy.flipud(numpy.arange(len(list2)))    
        
        totalScore = score1 + score2 
        
        return numpy.flipud(numpy.argsort(totalScore))
    
    @staticmethod
    def generateItemList(lists): 
        itemList = set([])

        for lst in lists: 
            itemList = itemList.union(set(lst))
        
        itemList = list(itemList)
        return itemList 
    
    @staticmethod 
    def generateTransitionMatrix(lst, itemList):     
        n = len(itemList)
        #Pj = scipy.sparse.lil_matrix((n, n))
        
        rowInds = array.array('i')      
        colInds = array.array('i')
        data = array.array('f')
        
        indexList = numpy.zeros(len(lst), numpy.int)            
        
        for i, item in enumerate(lst): 
            indexList[i] = itemList.index(item)
            
        for i in range(indexList.shape[0]): 
            validStates = indexList[0:i+1]
            rowInds.extend(numpy.ones(i+1, numpy.int)*indexList[i])
            colInds.extend(indexList[0:i+1])
            data.extend(numpy.ones(i+1)*1.0/validStates.shape[0])
            #Pj[indexList[i], validStates] = 1.0/validStates.shape[0]    
            
        Pj = scipy.sparse.csc_matrix((data, (rowInds, colInds)), shape=(n, n))
        return Pj
    
    @staticmethod 
    def computeOutputList(P, itemList):
        """
        Given a transition matrix, compute an ordering. 
        """        
        n = len(itemList)
        #If all lists agree on top elements then we get a stationary distribution 
        #of 1 for that index and zero elsewhere. Therefore add a little noise. 
        P += numpy.ones((n, n))*0.0001
        for i in range(n): 
            P[i, :] = P[i, :]/P[i, :].sum()
                
        u, v = scipy.sparse.linalg.eigs(P.T, 1)
        v = numpy.array(v).flatten()
        scores = numpy.abs(v)
        assert abs(u-1) < 0.001

        inds = numpy.flipud(numpy.argsort(scores)) 
        
        outputList = [] 
        for ind in inds: 
            outputList.append(itemList[ind])
            
        return outputList, scores 
    
    
    @staticmethod 
    def MC2(lists, itemList, alpha=None, verbose=False): 
        """
        Perform weighted rank aggregation using MC2 as given in Rank Aggregation Methods 
        for the Web, Dwork et al. The weighting vector is given by alpha. 
        
        :param lists: A list of lists. Each sublist is an ordered set of a subset of the items from itemList 
        
        :param itemList: A list of all possible items 
        """
        
        n = len(itemList)
        ell = len(lists)
        
        if alpha == None: 
            alpha = numpy.ones(ell)/ell
        
        P = numpy.zeros((n, n))
        PList = [] 
        
        logging.debug("Computing permutation matrices")
        for j, lst in enumerate(lists): 
            Util.printIteration(j, 1, ell)
            Pj = RankAggregator.generateTransitionMatrix(lst, itemList)

            P = P + alpha[j] * Pj 
            PList.append(Pj)
        
        P /= ell 
        logging.debug("Done")

        outputList,scores = RankAggregator.computeOutputList(P, itemList)
        
        if verbose: 
            return outputList, scores, PList
        else: 
            return outputList, scores
              
    @staticmethod    
    def supervisedMC2(lists, itemList, topQList): 
        """
        Use the supervised rank aggregation from the paper "Supervised Rank 
        Aggregation". 
        """
        outputList, scores, PList = RankAggregator.MC2(lists, itemList, verbose=True)
        ell = len(PList)
        n = len(itemList)        
        q = len(topQList)
        
        A = numpy.zeros((ell, n))
        for i in range(ell): 
            A[i, :] = numpy.diag(PList[i])
        
        notQList = list(set(itemList).difference(set(topQList)))    
        
        #Construct matrix H 
        H = numpy.zeros((q*(n-q), n))
        k = 0        
        
        for i in range(q): 
            for j in range(n-q): 
                H[itemList.indexof(topQList[i]), k] = -1 
                H[itemList.indexof(notQList[j]), k] = 1 
                k += 1 
        #Might need a way to limit the size of H 
        r = H.shape[0]
        
        zeroEllEll = numpy.zeros((ell, ell))
        zeroEllR =numpy.zeros((ell, ell)) 
        zeroNN = numpy.zeros((n, n))
        H00 = numpy.c_[zeroEllEll, -A, zeroEllR]
        H01 = numpy.c_[-A.T, zeroNN, numpy.zeros((n, r))]
        H02 = numpy.c_[numpy.zeros((m, ell)), numpy.zeros((r, n)), numpy.eye(r)]
        H0 = numpy.r_[H00, H01, H02]
        
        H10 = numpy.c_[-numpy.eye(ell), numpy.zeros((ell, n)), numpy.zeros((ell, r))]
        H11 = numpy.c_[-numpy.zeros((r, ell)), numpy.zeros((r, n)), numpy.eye(r)]
        H1 = numpy.r_[H10, H11]
        
        H2 = numpy.zeros((2, (ell+n+r)))
        H2[0, 0:ell]  = 1 
        H2[1, ell:(ell+n)] = 1
        
        H30 = numpy.c_[numpy.zeros((n, ell)), -numpy.eye(n), numpy.zeros((n, r))]
        H31 = numpy.c_[numpy.zeros((r, ell)), H, -numpy.eye(r)]
        H3 = numpy.r_[H30, H31]
        
        #We have vector rho = [lambda_0, lambda_1, lambda_2, lambda_3, gamma]
        c = numpy.zeros(ell+n+2*r+3)
        c[-1] = -1 
        
        Gl1 = numpy.zeros((ell+n+2*r+2, ell+n+2*r+3)) 
        numpy.put(Gl, numpy.arange(ell+n+2*r+2), numpy.arange(ell+n+2*r+2), 1)
        hl1 = numpy.zeros(ell+n+2*r+2)        
        
        #Finish later 
        #Gl2 = 
        
        
    @staticmethod 
    def supervisedMC22(lists, itemList, topQList, verbose=False): 
        """
        A supervised version of MC2 of our own invention. The idea is to find a 
        linear combination of transition matrices to fit a given one. 
        """
        import cvxopt
        import cvxopt.solvers
        ell = len(lists)
        n = len(itemList)
        outputList, scores, PList = RankAggregator.MC2(lists, itemList, verbose=True)
        
        Q = cvxopt.spmatrix([], [], [], (n*n, len(lists)))

        for i, P in enumerate(PList): 
            #print(P.todense())
            Q[:, i] = cvxopt.matrix(numpy.array(P.todense()).ravel()) 
            
        QQ = Q.T * Q
        
        Py = RankAggregator.generateTransitionMatrix(topQList, itemList)
        s = numpy.array(Py.todense()).ravel()
        s = cvxopt.matrix(s)
        
        G = cvxopt.spdiag((-numpy.ones(ell)).tolist())
        h = cvxopt.matrix(numpy.zeros(ell))
        
        A = cvxopt.matrix(numpy.ones(ell), (1, ell))
        b = cvxopt.matrix(numpy.ones(1))        
                
        q = -Q.T * s  
        
        sol = cvxopt.solvers.qp(QQ, q, G, h, A, b)
        
        alpha = numpy.array(sol['x'])
        
        #Combine the matrices 
        P = numpy.zeros((n, n))       
        
        for j, Pj in enumerate(PList): 
            Util.printIteration(j, 1, ell)
            P += alpha[j] * numpy.array(Pj.todense()) 

        P /= ell 
        
        outputList, scores = RankAggregator.computeOutputList(P, itemList)
        
        if verbose: 
            return outputList, scores, PList
        else: 
            return outputList, scores

    @staticmethod 
    def supervisedMC23(lists, itemList, topQList, verbose=False): 
        """
        A supervised version of MC2 of our own invention. The idea is to find a 
        linear combination of transition matrices to fit a given one. We just make
        sure it fits the stationary distribution. 
        """
        import cvxopt
        import cvxopt.solvers
        ell = len(lists)
        n = len(itemList)
        outputList, scores, PList = RankAggregator.MC2(lists, itemList, verbose=True)
        
        Py = RankAggregator.generateTransitionMatrix(topQList, itemList)
        u, v = scipy.sparse.linalg.eigs(Py.T, 1)
        v = numpy.array(v).flatten()

        c = numpy.zeros(v.shape[0])

        for i, P in enumerate(PList): 
            Q[:, i] = cvxopt.matrix(numpy.array(P.todense()).ravel()) 
            
        c = cvxopt.matrix(c)
        QQ = Q.T * Q
        
        Py = RankAggregator.generateTransitionMatrix(topQList, itemList)
        s = numpy.array(Py.todense()).ravel()
        s = cvxopt.matrix(s)
        
        G = cvxopt.spdiag((-numpy.ones(ell)).tolist())
        h = cvxopt.matrix(numpy.zeros(ell))
        
        A = cvxopt.matrix(numpy.ones(ell), (1, ell))
        b = cvxopt.matrix(numpy.ones(1))        
                
        q = -Q.T * s  
        
        sol = cvxopt.solvers.qp(QQ, q, G, h, A, b)
        
        alpha = numpy.array(sol['x'])
        
        #Combine the matrices 
        P = numpy.zeros((n, n))       
        
        for j, Pj in enumerate(PList): 
            Util.printIteration(j, 1, ell)
            P += alpha[j] * numpy.array(Pj.todense()) 

        P /= ell 
        
        outputList, scores = RankAggregator.computeOutputList(P, itemList)
        
        if verbose: 
            return outputList, scores, PList
        else: 
            return outputList, scores        
          
    @staticmethod 
    def greedyMC2(lists, itemList, trainList, n): 
        """
        A method to greedily select a subset of the outputLists such that 
        the average precision is maximised
        """
        currentListsInds = range(len(lists))
        newListsInds = []
        currentAvPrecision = 0 
        lastAvPrecision = -0.1
        
        while currentAvPrecision - lastAvPrecision > 0: 
            lastAvPrecision = currentAvPrecision 
            averagePrecisions = numpy.zeros(len(currentListsInds))      
            
            for i, j in enumerate(currentListsInds):
                newListsInds.append(j)

                newLists = []                
                for k in newListsInds: 
                    newLists.append(lists[k])
                
                rankAggregate, scores = RankAggregator.MC2(newLists, itemList)
                averagePrecisions[i] = Evaluator.averagePrecisionFromLists(trainList, rankAggregate[0:n], n)
                newListsInds.remove(j)

            j = numpy.argmax(averagePrecisions)
            currentAvPrecision = averagePrecisions[j]
            
            if currentAvPrecision > lastAvPrecision: 
                newListsInds.append(currentListsInds.pop(j))
            
        return newListsInds 
            
                
        
        