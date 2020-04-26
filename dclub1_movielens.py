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



numpartitions = 2 * 8
numAppearancesThresh = 15
maxNumRounds = 99876
alpha = 0.03
alpha2 = 0.7
sharing_delay = 5000
numAgents = 10
numShares = 1
init_rounds = 5000 #number of interactions
#path2='/Users/mahadik/online_clustering_bandits/source codes and data sets/'
path2='/home/hadoop/'
dataset ='movielens'


def cb(t1,t2):
    thresh = alpha2*( np.sqrt((1+np.log(1+t1))/(1+t1)) + np.sqrt((1+np.log(1+t2))/(1+t2)) );
    return thresh

def playUCBInd_series1(list_v,alpha,dim,numCtxts,Auser,buser,tuser):
    reward_l=[]
    
    for t in list_v:
        userID=np.where(userset==U_bc.value[t][0])[0][0]
        #A = covCBInd[userID]
        #b = bCBInd[userID]
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
        reward = R_bc.value[t,choice];
        Auser = Auser + np.dot(chosenCtxt,np.transpose(chosenCtxt))
        buser = buser + reward*chosenCtxt;
        tuser = tuser + 1
        reward_l=reward_l+[reward]
    return reward_l,Auser,buser,tuser

def reset_cluster_for_user_i(i):
        set1 = VCBInd[i]
        for j in range(0,numUsers):
            if (set1[j] == 1):
                ALoc = covCBInd[i]
                bLoc = bCBInd[i]
                ALocTemp = covCBInd[j]
                bLocTemp =  bCBInd[j]
                thetaLoc = np.dot(np.linalg.inv(ALoc),bLoc) 
                thetaLocTemp = np.dot(np.linalg.inv(ALocTemp),bLocTemp)
                if (np.linalg.norm(thetaLoc - thetaLocTemp) > cb(tCBInd[i],tCBInd[j])):
                    set1[j]=0
        return set1

def playCBmodified_series2(list_t,alpha,dim,numCtxts):  #tuser is tcluster
    reward_l=[]
    Acluster = covclusCBInd[list_t[0]]
    bcluster = bclusCBInd[list_t[0]]
    tuser = tclusCBInd[list_t[0]]
    update_l=[]
    
    for t in range(1,len(list_t)):
        userID=np.where(userset==U_bc.value[t][0])[0][0]
        #cid=CInd[userID]
        UCBs = np.zeros((numCtxts))
        allCtxts = X_bc.value[t,:,:]
        allCtxts=np.reshape(allCtxts, (allCtxts.shape[0],allCtxts.shape[1], 1)) 

        #A = covclusCBInd[cid]
        #b = bclusCBInd[cid]
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
        reward =  R_bc.value[t,choice];
        addcov =  np.dot(chosenCtxt,np.transpose(chosenCtxt))
        addb =  reward*chosenCtxt;
        #ALoc = covCBInd[userID]; 
        #bLoc = bCBInd[userID]
        #Auser = Auser + np.dot(chosenCtxt,np.transpose(chosenCtxt))
        #buser = buser + reward*chosenCtxt
        Acluster = Acluster + addcov
        bcluster =bcluster + addb
        tuser = tuser + 1  # Change to tcluster?
        reward_l=reward_l+[reward]
        update_l=update_l+[(addcov,addb,userID)]
    return reward_l,update_l


def playCBmodified_series(list_t,alpha,dim,numCtxts,tuser,Acluster,bcluster):  #tuser is tcluster
    reward_l=[]
    for t in list_t:
        userID=np.where(userset==U_bc.value[t][0])[0][0]
        #cid=CInd[userID]
        UCBs = np.zeros((numCtxts))
        allCtxts = X_bc.value[t,:,:]
        allCtxts=np.reshape(allCtxts, (allCtxts.shape[0],allCtxts.shape[1], 1)) 

        #A = covclusCBInd[cid]
        #b = bclusCBInd[cid]
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
        reward = R_bc.value[t,choice];
        #ALoc = covCBInd[userID]; 
        #bLoc = bCBInd[userID]
        #Auser = Auser + np.dot(chosenCtxt,np.transpose(chosenCtxt))
        #buser = buser + reward*chosenCtxt
        Acluster = Acluster+np.dot(chosenCtxt,np.transpose(chosenCtxt))
        bcluster =bcluster +reward*chosenCtxt;
        tuser = tuser + 1  # Change to tcluster?
        reward_l=reward_l+[(chosenCtxt,reward,userID)]
    return reward_l


def updateUser(users):
    covCBInd[users] = total_ans[users][1]
    bCBInd[users] =total_ans[users][2]
    tCBInd[users] = total_ans[users][3]


def updateUser_clusters(item):
        userID=item[0]
        for updates in item[1]:
            covCBInd[userID]=covCBInd[userID]+updates[0]
            bCBInd[userID] =  bCBInd[userID]+updates[1]
            tCBInd[userID]=tCBInd[userID]+1



t1=time.time()
M=pd.read_csv(path2+dataset+"_"+str(numAppearancesThresh)+"_userRoundIndcs.csv",header=None)

X=pd.read_csv(path2+dataset+"_X.csv",header=None)
U=pd.read_csv(path2+dataset+"_U.csv",header=None)
R=pd.read_csv(path2+dataset+"_R.csv",header=None)


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


t2=time.time()
print("Initialization time: "+str(t2-t1))


#New algorithm - Distributed CLUB
init_time=time.time()
covCBInd  = [np.eye(dim)] * (numUsers) 
bCBInd = [np.zeros((dim,1))] * (numUsers)
tCBInd = np.zeros((numUsers))


sc = SparkContext()
X_bc = sc.broadcast(X)
U_bc = sc.broadcast(U)
R_bc = sc.broadcast(R)



t1=time.time()


data = [(np.where(userset==U[M[0][i]-1][0])[0][0],M[0][i]-1) for i in range(0,init_rounds)]

dUser = collections.defaultdict(list)
key_count = collections.defaultdict(int)


for k, v in data:
    dUser[k].append(v)
    key_count[k]+=1

#print(key_count.items())

distData=sc.parallelize(dUser.items(),numpartitions)

total_ans=distData.map(lambda x: playUCBInd_series1(list(x[1]),alpha,dim,numCtxts,covCBInd[x[0]],bCBInd[x[0]],tCBInd[x[0]])) 

total_ans=total_ans.collect()

t2=time.time()
print("Spark functions Map time parallelize on number of users: "+str(t2-t1))

t1=time.time()

total_reward=[]
userlist = distData.keys().collect()

#make this parallel


p = Pool(processes=16)
p.map (updateUser, userlist)
p.close()  

for users in userlist:
    total_reward=total_reward+total_ans[users][0]
#    covCBInd[users] = total_ans[users][1]
#    bCBInd[users] =total_ans[users][2]
#    tCBInd[users] = total_ans[users][3]
t2=time.time()
print("Apply update time parallelize on number of users: "+str(t2-t1))


print(len(total_reward))
'''
t1=time.time()
for val in total_ans:
    for item in val:
        chosenCtxt=item[0]
        reward=item[1]
        userID=item[2]
        covCBInd[userID] = covCBInd[userID] + np.dot(chosenCtxt,np.transpose(chosenCtxt))
        bCBInd[userID] = bCBInd[userID] + reward*chosenCtxt
        tCBInd[userID] = tCBInd[userID] + 1
        total_reward=total_reward+[reward]
'''



t1=time.time()

num_cluscb = np.zeros((gamerounds))
VCBInd = np.ones((numUsers,numUsers))
for i in range(0,numUsers):
    VCBInd[i,i] = 0
CInd = np.zeros((numUsers))
[SInd,CInd]=connected_components(VCBInd, directed = False)
for i in range(0,init_rounds):
    num_cluscb[i] = SInd 

userData = [i for i in range(0,numUsers)]
userDist = sc.parallelize(userData,numpartitions)
userRecords = userDist.map(lambda x:  reset_cluster_for_user_i(x)).collect()

for i in range(0,numUsers):
    VCBInd[i]=userRecords[i]

[SInd,CInd]=connected_components(VCBInd, directed = False)
num_cluscb[init_rounds]=SInd

covclusCBInd  = [np.eye(dim)] * (SInd) 
bclusCBInd = [np.zeros((dim,1))] * (SInd)
tclusCBInd = np.zeros((SInd))


for i in range(0,numUsers):
    covclusCBInd[CInd[i]]=covclusCBInd[CInd[i]]+covCBInd[i] - np.eye(dim)
    bclusCBInd[CInd[i]] = bclusCBInd[CInd[i]] +  bCBInd[i]
    tclusCBInd[CInd[i]] = tclusCBInd[CInd[i]] + tCBInd[i]
t2=time.time()
print("Clustering time on number of users: "+str(t2-t1))

for repeat in range(0,1,1):
    for j in range(init_rounds,maxNumRounds,sharing_delay):
        T1=j
        T2=T1+sharing_delay

        t1=time.time()

        data = [(CInd[np.where(userset==U[M[0][i]-1][0])[0][0]],M[0][i]-1) for i in range(T1,T2)]
        dCluster = collections.defaultdict(list)
        key_count = collections.defaultdict(int)
        for k, v in data:
            dCluster[k].append(v)
            key_count[k]+=1

#resize clusters
        next_key=len(dCluster.keys())-1
        #d = collections.defaultdict(list)
        max_limit=5000

        for item in dCluster.copy().items():
            next_key=next_key+1
            if(len(item[1]) > max_limit):
                count=0
                for vals in dCluster[item[0]]:
                    if count >= max_limit:
                        next_key=next_key+1
                        count=1
                        dCluster[next_key].append(vals)
                    else:
                        count=count+1
                        dCluster[next_key].append(vals)
                    if(count==1):
                        dCluster[next_key].insert(0,item[0])
                del dCluster[item[0]]
            else:
                 dCluster[item[0]].insert(0,item[0])

        #print(dCluster.items())

        distData=sc.parallelize(dCluster.items(),numpartitions)
        total_ans=distData.map(lambda x: (playCBmodified_series2(list(x[1]),alpha,dim,numCtxts)))
        total_ans=total_ans.collect()
        t2=time.time()
        print("Spark Map time parallelize on number of clusters: "+str(t2-t1))


        #clusterlist = distData.keys().collect()

        #create new to old cluster mapping - not required if not updating


        #make this parallel
        #Is covclus and bclus used further?- if not then dont return
        #for clusters in clusterlist:
        #    covclusCBInd[clusters]= total_ans[clusters][1]
        #    bclusCBInd[clusters]= total_ans[clusters][2]
        #       addcov,addb,reward,userID


        t1=time.time()
        total_set_reward = []
        total_set_update = []

        for val in total_ans:
            total_set_reward=total_set_reward+ val[0]
            total_set_update=total_set_update +val[1]

        dUser = collections.defaultdict(list)
        for item in total_set_update:
            dUser[item[2]].append([item[0],item[1]])


        #for rewards in total_set_reward:
        total_reward=total_reward+total_set_reward

        p=Pool(16)
        p.map(updateUser_clusters,dUser.items());
        p.close()

        t2=time.time()
        print("Apply updates time parallelize on number of clusters: "+str(t2-t1))
        print(len(total_reward))
        print(len(total_ans))


        t1=time.time()
        userData = [i for i in range(0,numUsers)]
        userDist = sc.parallelize(userData,numpartitions)
        userRecords = userDist.map(lambda x:  reset_cluster_for_user_i(x)).collect()

        for i in range(0,numUsers):
            VCBInd[i]=userRecords[i]


        [SInd,CInd]=connected_components(VCBInd, directed = False)
        for i in range(T1,T2):
            num_cluscb[i] = SInd 

        covclusCBInd  = [np.eye(dim)] * (SInd) 
        bclusCBInd = [np.zeros((dim,1))] * (SInd)
        tclusCBInd = np.zeros((SInd))

        for i in range(0,numUsers):
                    covclusCBInd[CInd[i]]=covclusCBInd[CInd[i]]+covCBInd[i] - np.eye(dim)
                    bclusCBInd[CInd[i]] = bclusCBInd[CInd[i]] +  bCBInd[i]
                    tclusCBInd[CInd[i]] = tclusCBInd[CInd[i]] + tCBInd[i]

        t2=time.time()
        print("Clustering time on number of users: "+str(t2-t1))
        print("New clusters: "+str(SInd))


#print clusterid of user and translated time of interaction for next n rounds
sc.stop()
end_time=time.time()
dclub_time=end_time-init_time
print("Distributed CLUB took : "+str(dclub_time))


def playRandom(t):
    choice = np.random.choice(numCtxts);
    chosenCtxt = np.transpose(X[t,choice,:])
    reward = R[t,choice]
    return reward
    


RewardRandom = np.zeros((gamerounds))

t1=time.time()
for t in range(0,gamerounds):
    RewardRandom[t]= playRandom(M[0][t]-1)
t2=time.time()
print("Random play took seconds : "+str(t2-t1))


print(len(RewardRandom))
print(len(total_reward))

result = np.cumsum(total_reward)/np.cumsum(RewardRandom)
np.save("result_dclub",result)
print("Total_reward DClub , normalized by random:"+str(np.ma.masked_invalid(result).sum()))

np.save("clustering_dclub",num_cluscb)

