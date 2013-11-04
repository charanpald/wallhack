"""
This is just a class which represents a set of parameters for ABC model selection. 
"""
import numpy
import logging
import scipy.stats as stats
from apgl.util.Parameter import Parameter

class HIVABCParameters(object):
    def __init__(self, meanTheta, sigmaTheta, pertTheta, upperInfected=1500):
        """
        Initialised this object with a mean value of theta 
        """
        self.paramFuncs = []
        self.priorDists = []
        self.priorDensities = []
        self.perturbationKernels = []
        self.perturbationKernelDensities = []
        
        self.meanTheta = meanTheta
        self.sigmaTheta = sigmaTheta 
        self.pertTheta = pertTheta 
        self.upperInfected = upperInfected
        self.maxK = 10**5

        #Now set up all the parameters
        ind = 0 
        mu = meanTheta[ind]
        sigma = sigmaTheta[ind]
        priorDist, priorDensity = self.createDiscTruncNormParam(float(sigma), float(mu), upperInfected)
        perturbationKernel, perturbationKernelDensity = self.__createNormalDiscPert()
        self.__addParameter(("graph", "setRandomInfected"), priorDist, priorDensity, perturbationKernel, perturbationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = sigmaTheta[ind]
        priorDist, priorDensity = self.createTruncNormParam(sigma, mu)
        perturbationKernel, perturbationKernelDensity = self.__createNormalPert()
        self.__addParameter(("rates", "setAlpha"), priorDist, priorDensity, perturbationKernel, perturbationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = sigmaTheta[ind]
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        perturbationKernel, perturbationKernelDensity = self.__createNormalPert()
        self.__addParameter(("rates", "setRandDetectRate"), priorDist, priorDensity, perturbationKernel, perturbationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = sigmaTheta[ind]
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        perturbationKernel, perturbationKernelDensity = self.__createNormalPert()
        self.__addParameter(("rates", "setCtRatePerPerson"), priorDist, priorDensity, perturbationKernel, perturbationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = sigmaTheta[ind]
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        perturbationKernel, perturbationKernelDensity = self.__createNormalPert()
        self.__addParameter(("rates", "setContactRate"), priorDist, priorDensity, perturbationKernel, perturbationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = sigmaTheta[ind]
        priorDist, priorDensity = self.createTruncNormParam(sigma, mu)
        perturbationKernel, perturbationKernelDensity = self.__createNormalPert()
        self.__addParameter(("rates", "setInfectProb"), priorDist, priorDensity, perturbationKernel, perturbationKernelDensity)


    def createDiscTruncNormParam(self, sigma, mode, upper, lower=0):
        """
        Discrete truncated norm parameter 
        """
        Parameter.checkFloat(sigma, 0.0, float('inf'))
        Parameter.checkFloat(mode, 0.0, float('inf'))
        a = (lower-mode)/sigma
        b = (upper-mode)/sigma
        priorDist = lambda: round(stats.truncnorm.rvs(a, b, loc=mode, scale=sigma))
        priorDensity = lambda x: stats.truncnorm.pdf(x, a, b, loc=mode, scale=sigma)
        return priorDist, priorDensity 

    def createTruncNormParam(self, sigma, mode):
        """
        Truncated norm parameter between 0 and 1 
        """
        Parameter.checkFloat(sigma, 0.0, 1.0)
        Parameter.checkFloat(mode, 0.0, float('inf'))
        a = -mode/sigma
        b = (1-mode)/sigma
        priorDist = lambda: stats.truncnorm.rvs(a, b, loc=mode, scale=sigma)
        priorDensity = lambda x: stats.truncnorm.pdf(x, a, b, loc=mode, scale=sigma)
        return priorDist, priorDensity 

    def createGammaParam(self, sigma, mu):
        Parameter.checkFloat(sigma, 0.0, float('inf'))
        Parameter.checkFloat(mu, 0.0, float('inf'))

        if mu == 0.0:
            raise ValueError("Gamma distribution cannot have mean zero.")

        theta = sigma**2/mu
        k = mu/theta

        if k > self.maxK: 
            k == self.maxK 
            logging.warn("k for gamma distribution > " + str(self.maxK) + ", clipping")

        priorDist = lambda: stats.gamma.rvs(k, scale=theta)
        priorDensity = lambda x: stats.gamma.pdf(x, k, scale=theta)

        return priorDist, priorDensity 

    def __createNormalPert(self):
        perturbationKernel = lambda x, sigma: stats.norm.rvs(x, sigma)
        perturbationKernelDensity = lambda old, new, sigma: stats.norm.pdf(new, old, sigma)
        return perturbationKernel, perturbationKernelDensity

    def __createNormalDiscPert(self):
        perturbationKernel = lambda x, sigma: numpy.round(stats.norm.rvs(x, sigma))
        perturbationKernelDensity = lambda old, new, sigma: stats.norm.pdf(new, old, sigma)
        return perturbationKernel, perturbationKernelDensity

    def __addParameter(self, paramFunc, priorDist, priorDensity, perturbationKernel, perturbationKernelDensity):
        self.paramFuncs.append(paramFunc)
        self.priorDists.append(priorDist)
        self.priorDensities.append(priorDensity)
        self.perturbationKernels.append(perturbationKernel)
        self.perturbationKernelDensities.append(perturbationKernelDensity)

    def getParamFuncs(self):
        return self.paramFuncs

    def sampleParams(self):
        theta = []

        for priorDist in self.priorDists:
            theta.append(priorDist())
            
        theta = numpy.array(theta)
        return theta

    def priorDensity(self, theta, full=False):
        """
        Return an array of prior densities for the given theta. If full==False 
        then return an overall density. 
        """
        density = []

        for i in range(len(self.priorDensities)): 
            density.append(self.priorDensities[i](theta[i])) 
        
        density = numpy.array(density)        
        
        if full: 
            return density
        else: 
            return density.prod()
 

    def perturbationKernel(self, theta, pertTheta=None):
        """
        Find a pertubation of theta based on the same random distributions used 
        to generate theta. The std is given by pertTheta. 
        """
        if pertTheta==None: 
            pertTheta = self.pertTheta
        
        newTheta = []

        for i, perturbationKernel in enumerate(self.perturbationKernels):
            newTheta.append(perturbationKernel(theta[i], pertTheta[i]))
        
        newTheta = numpy.array(newTheta)
        return newTheta

    def perturbationKernelDensity(self, oldTheta, theta, pertTheta=None, full=False):
        """
        Find a pertubation probability density of theta based on the same random 
        distributions used to generate theta. The std is given by pertTheta. 
        """
        if pertTheta==None: 
            pertTheta = self.pertTheta
        density = []

        for i, perturbationKernelDensity in enumerate(self.perturbationKernelDensities):
            density.append(perturbationKernelDensity(oldTheta[i], theta[i], pertTheta[i]))

        density = numpy.array(density) 
        
        if full: 
            return density
        else: 
            return density.prod()

    def __getstate__(self): 
        odict = self.__dict__.copy() 
        del odict['paramFuncs']   
        del odict['priorDists']   
        del odict['priorDensities']   
        del odict['perturbationKernels']   
        del odict['perturbationKernelDensities']   
        
        return odict
        
    def __setstate__(self, dict):
        params = HIVABCParameters(dict["meanTheta"], dict["sigmaTheta"], dict["pertScale"], dict["upperInfected"])
        self.__dict__.update(dict)   
        self.paramFuncs = params.paramFuncs     
        self.priorDists = params.priorDists     
        self.priorDensities = params.priorDensities     
        self.perturbationKernels = params.perturbationKernels     
        self.perturbationKernelDensities = params.perturbationKernelDensities 
        