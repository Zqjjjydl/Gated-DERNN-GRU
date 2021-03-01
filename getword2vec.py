import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch
import numpy as np

word2idx=torch.load('./data/wordIdx.pt')['word2idx']
idx2word=torch.load('./data/wordIdx.pt')['idx2word']

word2vec=torch.load('./data/word2vec.pt')
idx2vec={}
print(word2idx["H.K."])
for (word,index) in word2idx.items():
    idx2vec[index]=word2vec[word]

torch.save(idx2vec,'./data/idx2vec.pt')
exit()
# trainSet=torch.load('./data/dataSet.pt')['trainSet']
# testSet=torch.load('./data/dataSet.pt')['testSet']
# word2vec={}
# with open('./word2vec/GoogleNews-vectors-negative300.txt','rb') as f:
#     line=f.readline()
#     index=0
#     while line:
#         # if index%1000==0:
#         if index==927:
#             print(line)
#             print(index)
#         index+=1
#         line=line.split()
#         if line[0] in word2idx:
#             word2vec[line[0]]=np.array(line[1:])
#         line=f.readline()

# torch.save(word2vec,"./data/word2vec.pt")
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True) 


word2vec={}
index=0
for word in word2idx:
    if index%100==0:
        print(index)
    index+=1
    if word in model:
        word2vec[word]=np.array(model[word])
    else:
        print(word)
        word2vec[word]=0.01*np.random.rand(300)

torch.save(word2vec,"./data/word2vec.pt")