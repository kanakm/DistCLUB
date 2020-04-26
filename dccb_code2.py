import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import json
from sklearn.preprocessing import normalize 
import sys
import glob, os
import math as m
from multiprocessing import Pool
from functools import partial

from scipy.sparse.csgraph import *
from scipy.sparse import csr_matrix

from pyspark import SparkContext
from operator import add
import random
import collections
def cb(t1,t2):
    thresh = alpha2*( np.sqrt((1+np.log(1+t1))/(1+t1)) + np.sqrt((1+np.log(1+t2))/(1+t2)) );
    return thresh
def buffThresh(t0):
    return 4*np.log(max(t0,1));



def Agentshare(currAgent, iR,Areceive,breceive,Areceivebuff,breceivebuff,Asend,bsend,Asendbuff,bsendbuff,Areceivelocal,breceivelocal,thetasend,thetareceive,edge,tsend,treceive):
    if(edge):
        if (np.linalg.norm(thetasend - thetareceive) >= cb(tsend,treceive)):
            Areceive = Areceivelocal
            breceive = breceivelocal

            Areceivebuff[0] = Areceivelocal
            breceivebuff[0] = breceivelocal
            edge = 0
        elif np.sum(VLocDCCB[currAgent]) == np.sum(VLocDCCB[iR]):
            Areceive = (Areceive + Asend)/2
            breceive = (breceive +bsend)/2
            if(Areceivebuff.shape[1]==Asendbuff.shape[1]):
                Areceivebuff = (Areceivebuff + Asendbuff)/2
                breceivebuff = (breceivebuff+bsendbuff)/2
            edge = 1
    return Areceive,breceive,Areceivebuff,breceivebuff,edge
def playDCCB(list_v,alpha,alpha2,dim,numCtxts,covLocDCCB,bLocDCCB,covDCCB,bDCCB,covBuffDCCB,bBuffDCCB,tagent):

    reward_l=[]
    
    for t in list_v:
        UCBs = np.zeros((numCtxts))
        allCtxts = X[t,:,:]
        allCtxts=np.reshape(allCtxts, (allCtxts.shape[0],allCtxts.shape[1], 1)) 

    
        A_inv = np.linalg.inv(covDCCB)
        theta = np.dot(A_inv, bDCCB) 
        beta = np.sqrt(dim*np.log((1+tagent))*np.sqrt(tagent) + 1);

        for a in range(0,numCtxts):
            conf1=  np.sqrt(beta*np.dot(np.transpose(allCtxts[a,:]),(np.dot(A_inv,allCtxts[a,:]))))
            conf = np.sqrt(tagent)*alpha*conf1[0][0]
            UCBs[a] = np.dot(np.transpose(allCtxts[a,:]),theta)[0][0] + conf;   
          
        choice = np.argmax(UCBs)
        chosenCtxt = (X[t,choice,:])
        chosenCtxt = np.reshape(chosenCtxt,(chosenCtxt.shape[0],1))
        #if chosenCtxt == R[t]:
        #    reward = 1
        #else:
        #    reward=0

        reward = R[t,choice]

        covBuffDCCB= np.concatenate((covBuffDCCB,np.zeros((dim,dim))),axis=1)
        bBuffDCCB = np.concatenate((bBuffDCCB,np.zeros((dim,1))),axis=0)
        
        end = covBuffDCCB.shape[1]
        covBuffDCCB[:,end-dim:end]=np.dot(chosenCtxt,np.transpose(chosenCtxt))
        bBuffDCCB[end-dim:end] = reward*chosenCtxt;
        ALoc = covLocDCCB; 
        bLoc = bLocDCCB
        covLocDCCB = ALoc + np.dot(chosenCtxt,np.transpose(chosenCtxt))
        bLocDCCB = bLoc + reward*chosenCtxt


        if tagent == 0:
            covBuffDCCB=np.delete(covBuffDCCB,np.s_[0:dim],axis=1)
            bBuffDCCB = np.delete(bBuffDCCB,np.s_[0:dim],axis=0)

        if bBuffDCCB.shape[0] > buffThresh(np.sum(tagent)):
            covDCCB = covDCCB + covBuffDCCB[:,0:dim]
            bDCCB = bDCCB + bBuffDCCB[0:dim]
            covBuffDCCB=np.delete(covBuffDCCB,np.s_[0:dim],axis=1)
            bBuffDCCB = np.delete(bBuffDCCB,np.s_[0:dim],axis=0)
     
        tagent = tagent + 1;
        reward_l=reward_l+[reward]

    return (reward_l,covBuffDCCB,bBuffDCCB,covLocDCCB,bLocDCCB,covDCCB,bDCCB)
  


#path2='/Users/mahadik/online_clustering_bandits/source codes and data sets/'
path2='/home/hadoop/'
dataset='movielens'
numAppearancesThresh = 50

t1=time.time()
M=pd.read_csv(path2+dataset+"_"+str(numAppearancesThresh)+"_userRoundIndcs.csv",header=None)



X = pd.read_csv(path2+dataset+"_X.csv",header=None)
U = pd.read_csv(path2+dataset+"_U.csv",header=None)
R = pd.read_csv(path2+dataset+"_R.csv",header=None)

maxNumRounds = 80000
numCtxts=R.shape[1]
dim=int(X.shape[1]/numCtxts)
result = pd.concat([X, U,R], axis=1)
result2=result.groupby(result.iloc[:,X.shape[1]]).filter(lambda x : len(x)>=numAppearancesThresh)
#result = result.sample(frac=1)
C1=result2.as_matrix()
U_unique=C1[...,X.shape[1]:X.shape[1]+1]
X1=result.as_matrix()
X2=X1[...,:X.shape[1]]
U=X1[...,X.shape[1]:X.shape[1]+1]
R=X1[...,X.shape[1]+1:]
X=np.reshape(X2, (X2.shape[0], numCtxts,dim)) 
gamerounds = min(X2.shape[0],maxNumRounds)
numUsers=np.unique(U_unique).shape[0]
userset=np.unique(U_unique)

t1=time.time()
numAgents = numUsers
sharing_delay =5000 # this equals buffer length
init_rounds=0
maxNumRounds = 80000
alpha = 0.03
alpha2 = 0.7
init_rounds = 0
numShares = 1
numpartitions = 16


tDCCB = np.zeros((numAgents)) 
covDCCB = [np.eye(dim)] * (numAgents)
bDCCB = [np.zeros((dim,1))] * (numAgents)
covBuffDCCB = [np.zeros((dim,dim))]* (numAgents)
bBuffDCCB = [np.zeros((dim,1))]* (numAgents)
covLocDCCB =[np.eye(dim)] * (numAgents)
bLocDCCB = [np.zeros((dim,1))] * (numAgents)
VLocDCCB = np.ones((numAgents,numAgents))
num_clusDCCB = np.zeros((gamerounds))
thetaDCCB = [np.zeros((dim,1))] * (numAgents)
for i in range(0,numAgents):
    VLocDCCB[i,i] = 0 
total_DCCB=[]
t2=time.time()
print("Initialization time: "+str(t2-t1))


init_time=time.time()
sc = SparkContext()
for j in range(init_rounds,gamerounds,sharing_delay):
    T1=j
    T2=T1+sharing_delay
    t1=time.time()

    
    
    data = [(np.where(userset==U[M[0][i]-1][0])[0][0],M[0][i]-1) for i in range(T1,T2)]
    #data = [(np.where(userset==U[i][0])[0][0],i) for i in range(T1,T2)]


    dAgent = collections.defaultdict(list)
    key_count = collections.defaultdict(int)
    for k, v in data:
        dAgent[k].append(v)
        key_count[k]+=1

    distData=sc.parallelize(dAgent.items(),numpartitions)

    total_ans=distData.map(lambda x: (playDCCB(list(x[1]),alpha,alpha2,dim,numCtxts,covLocDCCB[x[0]],bLocDCCB[x[0]],covDCCB[x[0]],bDCCB[x[0]],covBuffDCCB[x[0]],bBuffDCCB[x[0]],tDCCB[x[0]])))



    total_ans = total_ans.collect()

    t2=time.time()
    print("Spark functions Map time parallelize on number of users: "+str(t2-t1))

    t1=time.time()

    total_reward=[]
    agentlist = distData.keys().collect()

    for i in range(0,len(agentlist)):
        covBuffDCCB[agentlist[i]] = total_ans[i][1]
        bBuffDCCB[agentlist[i]]=total_ans[i][2]
        covLocDCCB[agentlist[i]] =total_ans[i][3]
        bLocDCCB[agentlist[i]] = total_ans[i][4]
        covDCCB[agentlist[i]] = total_ans[i][5]
        bDCCB[agentlist[i]] =  total_ans[i][6]
#p = Pool(processes=16)
#p.map (updateUser, userlist)
#p.close()  

    for agents in range(0,len(agentlist)):
        total_DCCB=total_DCCB+total_ans[agents][0]
    #    covCBInd[users] = total_ans[users][1]
    #    bCBInd[users] =total_ans[users][2]
    #    tCBInd[users] = total_ans[users][3]
    t2=time.time()
    print("Apply update time paralle;")

    t1=time.time()

    for i in range(0,numAgents):
        Aloc = covLocDCCB[i]
        bloc = bLocDCCB[i]
        thetaDCCB[i] = np.dot(np.linalg.inv(Aloc),bloc) 


    
    [SDCCB,CDCCB]=connected_components(VLocDCCB, directed = False)
    if(T2>=maxNumRounds):
        for i in range(T1,maxNumRounds):
            num_clusDCCB[i] = SDCCB
    else:
        for i in range(T1,T2):
            num_clusDCCB[i] = SDCCB 


    for k in range(0,numShares):
        cpAgents = [i for i in range(numAgents)]
        listAgents = [i for i in range(numAgents)]
        random.shuffle(listAgents)
        distData=sc.parallelize(cpAgents,numpartitions)
        total_ans=distData.map(lambda x: (Agentshare(x,listAgents[x],covDCCB[x],bDCCB[x],covBuffDCCB[x],
            bBuffDCCB[x],covDCCB[listAgents[x]],bDCCB[listAgents[x]],covBuffDCCB[listAgents[x]],
            bBuffDCCB[listAgents[x]],covLocDCCB[listAgents[x]],bLocDCCB[listAgents[x]],thetaDCCB[x],thetaDCCB[listAgents[x]],VLocDCCB[x][listAgents[x]],
            tDCCB[x],tDCCB[listAgents[x]]))).collect()

        for i in range(0,numAgents):
            covDCCB[i]=total_ans[i][0]
            bDCCB[i]=total_ans[i][1]
            covBuffDCCB[i]=total_ans[i][2]
            bBuffDCCB[i]=total_ans[i][3]
            VLocDCCB[i][listAgents[i]] = 0;
            VLocDCCB[listAgents[i]][i] = 0;
    
    t2=time.time()
    print(str(numShares)+" rounds of sharing take time "+str(t2-t1))

sc.stop()
end_time=time.time()
dccb_time = end_time-init_time
print("DCCB took seconds : "+str(dccb_time))


def playRandom(t):
    choice = np.random.choice(numCtxts);
    chosenCtxt = np.transpose(X[t,choice,:])
    #if chosenCtxt == R[t]:
    #        reward = 1
    #else:
    #        reward=0
    reward = R[t,choice]

    return reward
    
RewardRandom = np.zeros((gamerounds))

t1=time.time()
for t in range(0,gamerounds):
    RewardRandom[t]= playRandom(t)
t2=time.time()
print("Random play took seconds : "+str(t2-t1))

print(len(RewardRandom))
print(len(total_DCCB))

result = np.cumsum(total_DCCB)/np.cumsum(RewardRandom)
np.save("result_dccb_movielens",result)
print("Total_reward DCCB , normalized by random:"+str(np.ma.masked_invalid(result).sum()))

np.save("clustering_dccb",num_clusDCCB)
