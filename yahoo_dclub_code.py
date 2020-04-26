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

def playUCBInd_users(list_v,alpha,alpha2,dim,numCtxts,Auser,buser,tuser):
    reward_l=[]
    
    for t in list_v:
        userID=np.where(userset==U_bc.value[t][0])[0][0]
        A_inv = np.linalg.inv(Auser)
        theta = np.dot(A_inv, buser) 
        UCBs = np.zeros((numCtxts))
        allCtxts = X_bc.value[t,:,:]
        allCtxts=np.reshape(allCtxts, (allCtxts.shape[0],allCtxts.shape[1], 1)) 
        beta = np.sqrt(dim*np.log((1+tuser))*np.sqrt(tuser) + 1);
        for a in range(0,numCtxts):
            conf1=  np.sqrt(beta*np.dot(np.transpose(allCtxts[a,:]),(np.dot(A_inv,allCtxts[a,:]))))
            conf = np.sqrt(tuser)*alpha*conf1[0][0]
            UCBs[a] = np.dot(np.transpose(allCtxts[a,:]),theta)[0][0] + conf;   
        choice = np.argmax(UCBs)
        chosenCtxt = X_bc.value[t,choice,:]
        chosenCtxt = np.reshape(chosenCtxt,(chosenCtxt.shape[0],1))
        if chosenCtxt == R_bc.value[t]:
            reward = 1
        else:
            reward=0
        #reward = R_bc.value[t,choice];
        Auser = Auser + np.dot(chosenCtxt,np.transpose(chosenCtxt))
        buser = buser + reward*chosenCtxt;
        tuser = tuser + 1
        reward_l=reward_l+[reward]
    return reward_l,Auser,buser,tuser

def reset_cluster_for_user_i(i):
        set1 = VCBInd[i]
        for j in range(0,numUsers):
            if (set1[j] == 1):
                if (np.linalg.norm(thetaCBInd[i] - thetaCBInd[j]) > cb(tCBInd[i],tCBInd[j])):
                    set1[j]=0
        return set1



            

def playCBmodified_series2(list_v,alpha,alpha2,dim,numCtxts,Acluster,bcluster,tcluster,Auser,buser,tuser):  #tuser is tcluster
    reward_l=[]
    
    for t in list_v:
        userID=np.where(userset==U_bc.value[t][0])[0][0]
        #cid=CInd[userID]
        UCBs = np.zeros((numCtxts))
        allCtxts = X_bc.value[t,:,:]
        allCtxts=np.reshape(allCtxts, (allCtxts.shape[0],allCtxts.shape[1], 1)) 

        if(tuser > tcluster): # is this tcluster-mean
            A_inv = np.linalg.inv(Auser)
            theta = np.dot(A_inv, buser)
        else:
            A_inv = np.linalg.inv(Acluster)
            theta = np.dot(A_inv, bcluster) 


        beta = np.sqrt(dim*np.log((1+tuser))*np.sqrt(tuser) + 1);
        for a in range(0,numCtxts):
            conf1=  np.sqrt(beta*np.dot(np.transpose(allCtxts[a,:]),(np.dot(A_inv,allCtxts[a,:]))))
            conf = np.sqrt(tuser)*alpha*conf1[0][0]
            UCBs[a] = np.dot(np.transpose(allCtxts[a,:]),theta)[0][0] + conf;   
              
        choice = np.argmax(UCBs)
        chosenCtxt = (X_bc.value[t,choice,:])
        chosenCtxt = np.reshape(chosenCtxt,(chosenCtxt.shape[0],1))
        if chosenCtxt == R_bc.value[t]:
            reward = 1
        else:
            reward=0
        addcov =  np.dot(chosenCtxt,np.transpose(chosenCtxt))
        addb =  reward*chosenCtxt;
        
        Auser = Auser + np.dot(chosenCtxt,np.transpose(chosenCtxt))
        buser = buser + reward*chosenCtxt
        
        tuser = tuser + 1  # Change to tcluster?
        reward_l=reward_l+[reward]
    return reward_l,Auser,buser,tuser



def updateUser(users,total_ans):
    covCBInd[users] = total_ans[users][1]
    bCBInd[users] =total_ans[users][2]
    tCBInd[users] = total_ans[users][3]

def updateUser_clusters(item):
        userID=item[0]
        for updates in item[1]:
            covCBInd[userID]=covCBInd[userID]+updates[0]
            bCBInd[userID] =  bCBInd[userID]+updates[1]
            tCBInd[userID]=tCBInd[userID]+1
        return covCBInd[userID],bCBInd[userID],tCBInd[userID],userID
        

max_limit=25000000
numpartitions = 2 * 8
numAppearancesThresh = 100
maxNumRounds = 50000
alpha = 0.03
alpha2 = 0.7
sharing_delay = 2500
#path2='/Users/mahadik/online_clustering_bandits/source codes and data sets/'
path2='/home/hadoop/'
dataset ='yahoo'

t1=time.time()



X = pd.read_csv(path2+dataset+"_X.csv",header=None)
U = pd.read_csv(path2+dataset+"_U.csv",header=None)
R = pd.read_csv(path2+dataset+"_R.csv",header=None)

'''
X = pd.read_csv(path2+dataset+"_"+str(numAppearancesThresh)+"_X.csv",header=None)
U = pd.read_csv(path2+dataset+"_"+str(numAppearancesThresh)+"_U.csv",header=None)
R = pd.read_csv(path2+dataset+"_"+str(numAppearancesThresh)+"_R.csv",header=None)


numCtxts=R.shape[1]
dim=int(X.shape[1]/numCtxts)

X1=X.as_matrix()
X=np.reshape(X1, (X1.shape[0], numCtxts,dim)) 
U=U.as_matrix()
R=R.as_matrix()
userset=np.unique(U)
numUsers=userset.shape[0]
gamerounds = min(X.shape[0],maxNumRounds)
'''

numCtxts=51
dim=int(X.shape[1]/numCtxts)
result = pd.concat([X, U,R], axis=1)
result2=result.groupby(result.iloc[:,X.shape[1]]).filter(lambda x : len(x)>=numAppearancesThresh)
#result = result.sample(frac=1)
C1=result2.as_matrix()
U_unique=C1[...,X.shape[1]:X.shape[1]+1]
X1=result2.as_matrix()
X2=X1[...,:X.shape[1]]
U=X1[...,X.shape[1]:X.shape[1]+1]
R=X1[...,X.shape[1]+1:]
X=np.reshape(X2, (X2.shape[0], numCtxts,dim)) 
gamerounds = min(X2.shape[0],maxNumRounds)
numUsers=np.unique(U_unique).shape[0]
userset=np.unique(U_unique)


t2=time.time()
print("Initialization time: "+str(t2-t1))



#New algorithm - Distributed CLUB
init_time=time.time()
covCBInd  = [np.eye(dim)] * (numUsers) 
bCBInd = [np.zeros((dim,1))] * (numUsers)
thetaCBInd = [np.zeros((dim,1))] * (numUsers)
tCBInd = np.zeros((numUsers))


sc = SparkContext()
X_bc = sc.broadcast(X)
U_bc = sc.broadcast(U)
R_bc = sc.broadcast(R)


maxNumRounds =50000
iteration_length = 10000
init_rounds=0
control = 0.5
user_based = int(control *iteration_length)
cluster_based = int((1-control) *iteration_length)

total_reward = []
t1=time.time()

num_cluscb = np.zeros((gamerounds))
VCBInd = np.ones((numUsers,numUsers))


for j in range(init_rounds,maxNumRounds,iteration_length):
    T1=j
    T2=T1+user_based
    data = [(np.where(userset==U[i][0])[0][0],i) for i in range(T1,T2)]
    dUser = collections.defaultdict(list)
    key_count = collections.defaultdict(int)
    for k, v in data:
        dUser[k].append(v)
        key_count[k]+=1
    distData=sc.parallelize(dUser.items(),numpartitions)
    total_ans=distData.map(lambda x: playUCBInd_users(list(x[1]),alpha,alpha2,dim,numCtxts,covCBInd[x[0]],bCBInd[x[0]],tCBInd[x[0]])) 
    total_ans=total_ans.collect()
    t2=time.time()
    print("Spark functions Map time parallelize on number of users: "+str(t2-t1))

    t1=time.time()

    userlist = distData.keys().collect()
    for i in range(0,len(userlist)):
        covCBInd[userlist[i]] = total_ans[i][1]
        bCBInd[userlist[i]] =total_ans[i][2]
        tCBInd[userlist[i]] = total_ans[i][3]
    for i in range(0,len(userlist)):
        total_reward=total_reward+total_ans[i][0]
    t2=time.time()
    print("Apply update time parallelize on number of users: "+str(t2-t1))

    t1=time.time()

    for i in range(0,numUsers):
        Aloc = covCBInd[i]
        bloc = bCBInd[i]
        thetaCBInd[i] = np.dot(np.linalg.inv(Aloc),bloc) 
    t2=time.time()
    print(" update user vectors: "+str(t2-t1))


    userData = [i for i in range(0,numUsers)]
    userDist = sc.parallelize(userData,numpartitions)
    userRecords = userDist.map(lambda x:  reset_cluster_for_user_i(x)).collect()
    for i in range(0,numUsers):
        VCBInd[i]=userRecords[i]
    [SInd,CInd]=connected_components(VCBInd, directed = False)
    covclusCBInd  = [np.eye(dim)] * (SInd) 
    bclusCBInd = [np.zeros((dim,1))] * (SInd)
    tclusCBInd = np.zeros((SInd))
    for i in range(0,numUsers):
        covclusCBInd[CInd[i]]=covclusCBInd[CInd[i]]+covCBInd[i] - np.eye(dim)
        bclusCBInd[CInd[i]] = bclusCBInd[CInd[i]] +  bCBInd[i]
        tclusCBInd[CInd[i]] = tclusCBInd[CInd[i]] + tCBInd[i]
    t2=time.time()
    print("Clustering time on number of users: "+str(t2-t1))

    T1=T2
    T2=T1+cluster_based
    #data = [(CInd[np.where(userset==U[M[0][i]-1][0])[0][0]],M[0][i]-1) for i in range(T1,T2)]
    
    data = [(CInd[np.where(userset==U[i][0])[0][0]],i) for i in range(T1,T2)]
    dCluster = collections.defaultdict(list)
    key_count = collections.defaultdict(int)
    for k, v in data:
        dCluster[k].append(v)
        key_count[k]+=1

    data = [(np.where(userset==U[i][0])[0][0],i) for i in range(T1,T2)]
    dUser = collections.defaultdict(list)
    for k, v in data:
        dUser[k].append(v)
        
    
    distData=sc.parallelize(dUser.items(),numpartitions)
    total_ans=distData.map(lambda x: (playCBmodified_series2(list(x[1]),alpha,alpha2,dim,numCtxts,covclusCBInd[CInd[x[0]]],bclusCBInd[CInd[x[0]]],(tclusCBInd[CInd[x[0]]]/key_count[CInd[x[0]]]),covCBInd[x[0]],bCBInd[x[0]],tCBInd[x[0]])))
    total_ans=total_ans.collect()
    t2=time.time()
    print("Spark Map time parallelize on number of clusters: "+str(t2-t1))

    t1=time.time()
    userlist = distData.keys().collect()
    for i in range(0,len(userlist)):
        covCBInd[userlist[i]] = total_ans[i][1]
        bCBInd[userlist[i]] =total_ans[i][2]
        tCBInd[userlist[i]] = total_ans[i][3]
    for i in range(0,len(userlist)):
        total_reward=total_reward+total_ans[i][0]
    t2=time.time()
    print("Apply update time parallelize on number of users: "+str(t2-t1))


      

#print clusterid of user and translated time of interaction for next n rounds
sc.stop()
end_time=time.time()
dclub_time=end_time-init_time
print("Distributed CLUB took : "+str(dclub_time))


def playRandom(t):
    choice = np.random.choice(numCtxts);
    chosenCtxt = np.transpose(X[t,choice,:])
    if chosenCtxt == R[t]:
            reward = 1
    else:
            reward=0
    return reward
    


RewardRandom = np.zeros((gamerounds))

t1=time.time()
for t in range(0,gamerounds):
    RewardRandom[t]= playRandom(t)
t2=time.time()
print("Random play took seconds : "+str(t2-t1))


print(len(RewardRandom))
print(len(total_reward))

result = np.cumsum(total_reward)/np.cumsum(RewardRandom)


np.save("result_dclub",result)
print("Total_reward DClub , normalized by random:"+str(np.ma.masked_invalid(result).sum()))

np.save("clustering_dclub",num_cluscb)



