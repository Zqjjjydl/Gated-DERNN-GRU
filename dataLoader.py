from torch.utils.data import Dataset
import torch

class trainDataset(Dataset): 
    def __init__(self):
        super(trainDataset,self).__init__()
        trainSet=torch.load('./data/trainSet.pt')
        self.tree=trainSet["trainSetTree"]
        self.sentenceInIndex=trainSet["trainSet_Index"]
        self.documentTopicVec=trainSet["trainSet_DocumentTopicVec"]
        self.target=trainSet["trainSetTarget"]
        dataClean=[148,308,717,1136,1150,1622,2527,2550,2623,3526,4078]
        self.tree=[self.tree[i] for i in range(len(self.tree)) if i not in dataClean]
        self.sentenceInIndex=[self.sentenceInIndex[i]\
             for i in range(len(self.sentenceInIndex)) if i not in dataClean]
        self.documentTopicVec=[self.documentTopicVec[i]\
             for i in range(len(self.documentTopicVec)) if i not in dataClean]
        self.target=[self.target[i] for i in range(len(self.target)) if i not in dataClean]


        for index in range(0,len(self.documentTopicVec)):
            self.documentTopicVec[index]=torch.tensor(self.documentTopicVec[index])

        
    def __getitem__(self, index):
        return self.tree[index],self.sentenceInIndex[index]\
            ,self.documentTopicVec[index],self.target[index]
    def __len__(self): 
        return len(self.tree)

class testDataset(Dataset): 
    def __init__(self):
        super(testDataset,self).__init__()
        testSet=torch.load('./data/testSet.pt')
        self.tree=testSet["testSetTree"]
        self.sentenceInIndex=testSet["testSet_Index"]
        self.documentTopicVec=testSet["testSet_DocumentTopicVec"]
        self.target=testSet["testSetTarget"]
        dataClean=[453,1043,1080,1895,1911]
        self.tree=[self.tree[i] for i in range(len(self.tree)) if i not in dataClean]
        self.sentenceInIndex=[self.sentenceInIndex[i]\
             for i in range(len(self.sentenceInIndex)) if i not in dataClean]
        self.documentTopicVec=[self.documentTopicVec[i]\
             for i in range(len(self.documentTopicVec)) if i not in dataClean]
        self.target=[self.target[i] for i in range(len(self.target)) if i not in dataClean]

        for index in range(0,len(self.documentTopicVec)):
            self.documentTopicVec[index]=torch.tensor(self.documentTopicVec[index])


        
    def __getitem__(self, index):
        return self.tree[index],self.sentenceInIndex[index]\
            ,self.documentTopicVec[index],self.target[index]
    def __len__(self): 
        return len(self.tree)