import pickle
import csv
from sandbox.util.PathDefaults import PathDefaults 

"""
Check the data is correct. 
"""

dataDir = PathDefaults.getDataDir()
contactsFilename = PathDefaults.getDataDir() + "reference/contacts_anonymised.tsv"
inputFileName = dataDir + "reference/author_document_count"

authorFile = open(inputFileName, 'rb')
csvReader = csv.reader(authorFile, delimiter='\t')

authorSet = set([])

for row in csvReader:
    authorSet.add(row[1])
    
print(len(authorSet))


coauthorsFile = open(contactsFilename, 'rb')
coauthorsReader = csv.reader(coauthorsFile, delimiter='\t')

contactsSet = set([])

for row in coauthorsReader:
    contactsSet.add(row[0])
    contactsSet.add(row[1])
    
print(len(contactsSet))

print(len(authorSet.intersection(contactsSet)))


#Check indexer 
authorIndexerFilename = dataDir + "reference/authorIndexerKeyword.pkl"   
authorIndexer = pickle.load(open(authorIndexerFilename))

authorSet = set(authorIndexer.getIdDict().keys())


print(len(authorSet), len(authorSet.intersection(contactsSet)))

#Check indexer 
authorIndexerFilename = dataDir + "reference/authorIndexerDoc.pkl"   
authorIndexer = pickle.load(open(authorIndexerFilename))

authorSet = set(authorIndexer.getIdDict().keys())

print(len(authorSet), len(authorSet.intersection(contactsSet)))
