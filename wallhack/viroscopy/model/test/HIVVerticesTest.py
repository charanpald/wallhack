

import apgl
import numpy 
import unittest
import pickle 
import numpy.testing as nptst 

from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVVertices import HIVVertices

@apgl.skipIf(not apgl.checkImport('sppy'), 'No module pysparse')
class  HIVGraphTest(unittest.TestCase):
    def setup(self):
        pass

    def testContructor(self):
        numVertices = 10
        vertices = HIVVertices(numVertices)

        
        self.assertEquals(numVertices, vertices.getNumVertices())
        self.assertEquals(8, vertices.getNumFeatures())
        
        
        
    def testAddVertices(self):
        numVertices = 10
        vertices = HIVVertices(numVertices)
        
        v1 = vertices[4]
        
        vertices.addVertices(5)
        print(vertices.V.shape)
        self.assertEquals(numVertices+5, vertices.getNumVertices())
        nptst.assert_array_equal(v1, vertices[4])
        

if __name__ == '__main__':
    unittest.main()
