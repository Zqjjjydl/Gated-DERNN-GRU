import torch
import torch.nn as nn
import torch.nn.functional as F

class DERNN(nn.Module):

    def __init__(self,hiddenDim,embDim,idx2vec):
        super(DERNN, self).__init__()
        self.q2idx={}
        self.q2idx=self.buildQ()
        self.idx2vec=idx2vec
        self.q=torch.nn.Parameter(torch.zeros(10,hiddenDim))#9+1
        self.W=torch.nn.Parameter(torch.zeros(hiddenDim*3,embDim))
        self.U=torch.nn.Parameter(torch.zeros(hiddenDim*3,hiddenDim))
        self.D=torch.nn.Parameter(torch.zeros(hiddenDim*3,hiddenDim))
        self.b=torch.nn.Parameter(torch.zeros(hiddenDim*3))
        

        self.hiddenDim=hiddenDim
        self.embDim=embDim
    def buildQ(self):
        depType=torch.load('./data/depSet.pt')
        q2idx={}
        for (index,dep) in enumerate(depType):
            q2idx[index]=dep
        length=len(q2idx)
        q2idx['leafNotion']=length
        return q2idx

    def forward(self,dpTree,wordIdx,sentenceInIndex):
        x_n=self.idx2vec[(sentenceInIndex[wordIdx-1]).item()]

        if wordIdx not in dpTree:
            rootChildList=[('leafNotion')]
            q_stack=torch.zeros(1,self.hiddenDim)
            q_stack[0]=self.q[-1]
            q_sum=torch.sum(q_stack,dim=0)

            iu=self.W[self.hiddenDim:,:].mm(x_n.view(-1,1))\
                +self.D[self.hiddenDim:,:].mm(q_sum.view(-1,1))\
                +(self.b[(self.hiddenDim):]).view(-1,1)
            
            i=torch.sigmoid(iu[:self.hiddenDim])
            u=torch.tanh(iu[self.hiddenDim:])
            h_n=torch.tanh(i*u)
            return h_n.view(-1)
            
        
        rootChildList=dpTree[wordIdx]
        h_stack=torch.zeros(self.hiddenDim,len(rootChildList))#hiddenDim,childnum
        q_stack=torch.zeros(self.hiddenDim,len(rootChildList))#hiddenDim,childnum
        xn_stack=torch.cat([x_n.view(-1,1) for i in range(len(rootChildList))],1)
        for (index,child) in enumerate(rootChildList):
            hi=self.forward(dpTree,child[0].item(),sentenceInIndex)
            h_stack[:,index]=hi
            q_stack[:,index]=self.q[child[1]]

        wf=self.W[:self.hiddenDim,:]#hiddenDim,embDim
        # wf=torch.cat([w_f for i in range(0,len(rootChildList))],0)#childnum*hiddenDim,embDim

        uf=self.U[:self.hiddenDim,:]#hiddenDim,hiddenDim
        # uf=torch.cat([u_f for i in range(0,len(rootChildList))],0)#childnum*hiddenDim,hiddenDim

        df=self.D[:self.hiddenDim,:]#hiddenDim,hiddenDim
        # df=torch.cat([d_f for i in range(0,len(rootChildList))],0)#childnum*hiddenDim,hiddenDim

        b_f=self.b[:self.hiddenDim]
        bf=torch.cat([b_f.view(-1,1) for i in range(0,len(rootChildList))],1)#hiddenDim,childnum

        # (hiddenDim,childNum)       (hiddenDim,childNum)      (hiddenDim,childNum)   
        f=wf.mm(xn_stack)+uf.mm(h_stack)+df.mm(q_stack)+bf#(hiddenDim,childNum)
        f=torch.sigmoid(f)#(hiddenDim,childNum)
        f=f*h_stack#(hiddenDim,childNum)
        f=torch.sum(f,dim=1)

        h_sum=torch.sum(h_stack,dim=1).view(-1,1)#hiddenDim,1
        q_sum=torch.sum(q_stack,dim=1).view(-1,1)#hiddenDim,1

        iu=self.W[self.hiddenDim:,:].mm(x_n.view(-1,1))\
            +self.U[self.hiddenDim:,:].mm(h_sum.view(-1,1))\
            +self.D[self.hiddenDim:,:].mm(q_sum.view(-1,1))\
                +(self.b[self.hiddenDim:]).view(-1,1)
        
        i=torch.sigmoid(iu[:self.hiddenDim]).view(-1)
        u=torch.tanh(iu[self.hiddenDim:]).view(-1)

        h_n=torch.tanh(i*u+f)

        return h_n
    def weightIni(self):
        nn.init.xavier_uniform_(self.q)
        nn.init.orthogonal_(self.W)
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.D)
        nn.init.normal_(self.b)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        pass

class GATE(nn.Module):
    def __init__(self,hiddenDim):
        super(GATE, self).__init__()
        self.W=torch.nn.Parameter(torch.zeros(hiddenDim*2,hiddenDim))
        self.U=torch.nn.Parameter(torch.zeros(hiddenDim*2,hiddenDim))
        self.b=torch.nn.Parameter(torch.zeros(hiddenDim*2))
        self.hiddenDim=hiddenDim

    def forward(self,d,t):
        g=self.W.mm(d)+self.U.mm(t)+(self.b).view(-1,1)
        gd=torch.sigmoid(g[:self.hiddenDim])
        gt=torch.sigmoid(g[self.hiddenDim:])
        v=torch.tanh(gd*d+gt*t)
        return v
    def weightIni(self):
        nn.init.orthogonal_(self.W)
        nn.init.orthogonal_(self.U)
        nn.init.normal_(self.b)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        pass


class Gated_DERNN_GRU_TOPIC(nn.Module):

    def __init__(self,hiddenDim,embDim,topicNum,targetNum,idx2vec):
        super(Gated_DERNN_GRU_TOPIC, self).__init__()
        self.DERNN=DERNN(hiddenDim,embDim,idx2vec)
        self.GRU=nn.GRU(hiddenDim,hiddenDim,batch_first=True)
        self.GATE=GATE(hiddenDim)
        self.MLP=nn.Linear(topicNum,hiddenDim)
        self.hidden2target=nn.Linear(hiddenDim,targetNum)
        self.idx2vec=idx2vec
        
        self.hiddenDim=hiddenDim
        self.embDim=embDim
        self.topicNum=topicNum
        self.targetNum=targetNum



    def forward(self,tree,sentenceInIndex,D_T_distr):
        sentenceNum=len(tree)
        sentenceVector=torch.zeros(1,sentenceNum,self.hiddenDim)
        for (index,sentence) in enumerate(tree):
            sentenceVector[0][index]=self.DERNN(sentence,sentence[0][0][0].item(),sentenceInIndex[index])
        h_0 = torch.randn(1,1,self.hiddenDim)
        documentVector,hiddenState=self.GRU(sentenceVector,h_0)
        topicVector=torch.tanh(self.MLP(D_T_distr))
        hiddenState=hiddenState.view(-1,1)
        topicVector=topicVector.view(-1,1)
        documentRepre=(self.GATE(hiddenState,topicVector)).view(-1)
        target=F.softmax(self.hidden2target(documentRepre))
        return target
    def weightIni(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        self.DERNN.weightIni()
        self.GATE.weightIni()


