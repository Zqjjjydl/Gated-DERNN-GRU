import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch
import torch.nn as nn
from network import Gated_DERNN_GRU_TOPIC
from tqdm import tqdm
from dataLoader import trainDataset
from dataLoader import testDataset
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


#--------------
# root =os.getcwd()+ '/../'

max_epoch=10
learning_rate=0.001
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
TOPIC_NUM=10
TAG_SIZE= 7#1-7
BATCH_SIZE=20
idx2vec=torch.load('./data/idx2vec.pt')
for key in idx2vec:
    idx2vec[key]=torch.tensor(idx2vec[key]).float()

#--------------

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

trainDataDealer = trainDataset()
testDataDealer = testDataset()
train_loader = DataLoader(dataset=trainDataDealer,
                          batch_size=1,
                          shuffle=False)

test_loader = DataLoader(dataset=testDataDealer,
                         batch_size=1,
                          shuffle=False)

progressive = tqdm(range(max_epoch), total=max_epoch,
                   ncols=50, leave=False, unit="b")

model=Gated_DERNN_GRU_TOPIC(HIDDEN_DIM,EMBEDDING_DIM,TOPIC_NUM,TAG_SIZE,idx2vec)
model.weightIni()
model=model.to(device)

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=[0.9,0.999])
loss_fn = F.kl_div 
loss_list=[]

for step in progressive:
    #train
    model.train()
    sentence_count=0
    model.zero_grad()
    for i, data in enumerate(test_loader):
        print(i)

        tree,sentenceInIndex\
            ,documentTopicVec,target=data
        documentTopicVec=documentTopicVec.to(device)
        targetVector=torch.zeros(TAG_SIZE)
        targetVector[target-1]=1
        
        predict=model(tree,sentenceInIndex,documentTopicVec)
        
        loss=loss_fn(predict.log(),targetVector, reduction='mean')
        with open("./result/log.txt","a") as f:
            f.write("{:.6f}\n".format(loss.item()))
        loss_list.append(loss.item())
        loss.backward()
        if (i%BATCH_SIZE)==0:
            optimizer.step()
            model.zero_grad()

    with open("./result/log.txt","a") as f:
        print(loss_list,file=f)
    torch.save(model.state_dict(),'./model/model1.pt')
print(loss_list)
with open("./result/log.txt","a") as f:
    print(loss_list,file=f)
