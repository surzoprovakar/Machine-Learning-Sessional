# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:22:31 2019

@author: surzo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from scipy.stats import multivariate_normal
eps=np.finfo(float).eps

def rmse(data1,data2):
    val=data1-data2
    rmse=0.0
    for i in range(len(data1)):
        for j in range (len(data1[0])):
            if(data1[i][j]!=99):
                rmse=rmse+val[i][j]*val[i][j]
    rmse=rmse/(len(data1)*len(data1[0]))
    rmse=np.sqrt(rmse)
    
    return np.sqrt(rmse)  

data=pd.read_csv('data.csv',header=None)
data_length=len(data)
#print(data_length)

split = 0.1*len(data)
split=int(split)
print(split)
data_small=data.iloc[0:split]

chng=data_small
chng2=chng.drop(chng.columns[0], axis=1)
rows,columns=chng2.shape
#print(type(chng2))
a=chng2.values
#print(type(a))
train=np.full((rows,columns),99)
valid=np.full((rows,columns),99)
test=np.full((rows,columns),99)
trn_val=np.full((rows,columns),99)

for i in range(rows):
    rat=[]
    for j in range(columns):
        if a[i][j]!=99:
            rat.append(j)
    t=0.6*len(rat)
    t=int(t)
    v=0.2*len(rat)
    v=int(v)
    tt=len(rat)-t-v
    for j in range(t):
        train[i][rat[j]]=a[i][j]
    for j in range(t,t+v):
        valid[i][rat[j]]=a[i][j]
    for j in range(t+v,t+v+tt):
        test[i][rat[j]]=a[i][j]
    for j in range(t+v):
        trn_val[i][rat[j]]=a[i][j]

out=open("out.txt","w")
lamda_u=[0.01,0.1,1.0,10.0]
lamda_v=[0.01,0.1,1.0,10.0]
K=[5,10,20,40]

error=1000000

for u in lamda_u:
    for v in lamda_v:
        for k in K:
            U=np.zeros((k,len(train)))
            V=np.zeros((k,len(train[0])))
            for i in range(k):
                for j in range(len(train)):
                    rand=np.random.uniform(-10,10)
                    U[i][j]=rand
            
            er=100000
            while 1:
                
                for i in range(len(train[0])):
                    sigma=[]
                    for j in range(len(train)):
                        if train[j][i]!=99:
                            sigma.append(j)
                    val=np.zeros((k,1))
                    val2=np.zeros((k,k))
                    
                    for j in range(len(sigma)):
                        a=np.zeros((k,1))
                        for m in range(k): 
                            a[m][0]=U[m][sigma[j]]
                        
                        val=val+train[j][i]*a            
                        x=a
                        y=np.transpose(x)
                        val2=val2+np.matmul(x,y)
                    idn=np.identity(k)
                    mul=v*idn
                    res=val2+mul
                    res2=np.linalg.inv(res)
                    vm=np.matmul(res2,val)
                    for m in range(k):
                        V[m][i]=vm[m][0]
                
                
                for i in range(len(train)):
                    sigma=[]
                    for j in range(k):
                        if train[i][j]!=99:
                            sigma.append(j)
                    val=np.zeros((k,1))
                    val2=np.zeros((k,k))
                    for j in range(len(sigma)):
                        a=np.zeros((k,1))
                        for m in range(k):
                            a[m][0]=V[m][sigma[j]]
                        val=val+train[i][j]*a
                        x=a
                        y=np.transpose(x)
                        val2=val2+np.matmul(x,y)
                    idn=np.identity(k)
                    mul=u*idn
                    res=val2+mul
                    res2=np.linalg.inv(res)
                    vm=np.matmul(res2,val)
                    for m in range(k):
                        U[m][i]=vm[m][0]
                    
                UU=np.transpose(U)
                chk=np.matmul(UU,V)
                rs=rmse(train,chk)
                
                if abs(rs-er)<2:
                    print(rs)
                    break
                else:
                    er=rs
            out.write('Lamda U=%f\n' %u)
            out.write('Lamda V=%f\n' %v)
            out.write('K=%f\n' %k)
            np.savetxt(out,UU,fmt='%.2f')
            out.write('\n')
            np.savetxt(out,V,fmt='%.2f')
            out.write('\n')
                    
            VU=np.transpose(U)
            chkv=np.matmul(VU,V)
            rsv=rmse(valid,chkv)
            if rsv<error:
                error=rsv
                fu=u
                fv=v
                fk=k
                fU=U
                fV=V
            
out.close()

err=100000
while 1:
    for i in range(len(trn_val[0])):
        sigma=[]
        for j in range(len(trn_val)):
            if trn_val[j][i]!=99:
                sigma.append(j)
        val=np.zeros((fk,1))
        val2=np.zeros((fk,fk))
                    
        for j in range(len(sigma)):
            a=np.zeros((fk,1))
            for m in range(fk): 
                a[m][0]=fU[m][sigma[j]]
                        
            val=val+trn_val[j][i]*a            
            x=a
            y=np.transpose(x)
            val2=val2+np.matmul(x,y)
        idn=np.identity(fk)
        mul=fv*idn
        res=val2+mul
        res2=np.linalg.inv(res)
        vm=np.matmul(res2,val)
        for m in range(fk):
            fV[m][i]=vm[m][0]
                
                
    for i in range(len(trn_val)):
        sigma=[]
        for j in range(fk):
            if trn_val[i][j]!=99:
                sigma.append(j)
        val=np.zeros((fk,1))
        val2=np.zeros((fk,fk))
        for j in range(len(sigma)):
            a=np.zeros((fk,1))
            for m in range(fk):
                a[m][0]=fV[m][sigma[j]]
            val=val+trn_val[i][j]*a
            x=a
            y=np.transpose(x)
            val2=val2+np.matmul(x,y)
        idn=np.identity(fk)
        mul=fu*idn
        res=val2+mul
        res2=np.linalg.inv(res)
        vm=np.matmul(res2,val)
        for m in range(fk):
            fU[m][i]=vm[m][0]
    
    ffU=np.transpose(fU)
    chk=np.matmul(ffU,fV)
    rs=rmse(trn_val,chk)
                
    if abs(rs-err)<1:
        #print(rs)
        lU=fU
        lV=fV
        break
    else:
        err=rs
            
  
llU=np.transpose(lU)
rms=np.matmul(llU,lV)
rs=rmse(test,rms)

print('final error ',rs)
final=np.full((rows,columns),99)
for i in range(len(test)):
    for j in range (len(test[0])):
        if(test[i][j]!=99):
            final[i][j]=test[i][j]-rms[i][j]

print(final)                
    

            
                
        
    
              
                    
