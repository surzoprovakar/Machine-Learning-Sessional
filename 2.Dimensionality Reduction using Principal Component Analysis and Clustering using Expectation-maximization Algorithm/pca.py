# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 23:12:49 2018

@author: Raju
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from scipy.stats import multivariate_normal
eps=np.finfo(float).eps

def max_num(num1,num2,num3):
    
    if (num1 >= num2) and (num1 >= num3):
        largest = num1
    elif (num2 >= num1) and (num2 >= num3):
        largest = num2
    else:
        largest = num3
        
    return largest

def Read_data():
    
    data=pd.read_csv('data.txt',sep="\t",header=None)
    data_length=len(data)
    #print(data_length)
    column=len(data.columns)
    #print(column)
    
    
    #b=np.full((data_length,data_length),1)
    #c=data-np.matmul(b,data)*1/data_length
    #var=np.matmul(np.transpose(c),c)*1/data_length
    
    ar=data.values
    arr=np.transpose(ar)
    cov=np.cov(arr)
    
    eigen_val, eigen_vec = np.linalg.eig(cov)
    eig_pair = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_val))]
    eig_pair.sort(reverse=True)
    
    max_two = np.hstack((eig_pair[0][1].reshape(column,1), eig_pair[1][1].reshape(column,1)))
    #print('Matrix W:\n', matrix_w)
    #transformed = matrix_w.T.dot(arr)
    max_two=np.transpose(max_two)
    plot=np.matmul(max_two,arr)
    plot=np.transpose(plot)
    #print(plot[0])
    x=np.array(0)
    y=np.array(0)
    for i in range(len(plot)):
        x=np.append(x,plot[i][0])
        y=np.append(y,plot[i][1])
    
    #plt.plot(plot[0,0:100], plot[1,0:100], 'o', markersize=6, color='red', label='class1')
    #plt.plot(plot[0,100:200], plot[1,100:200], '^', markersize=6, color='green', label='class2')
    plt.scatter(x,y,marker='o', linewidths=3,color='green')
    plt.title('Projecting data')
    
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()
    
    w=np.array([1/3,1/3,1/3])
    #print(w[0])
    single=np.matrix([[1,0.1],[0.1,1]])
    co=[]
    for i in range(len(w)):
        co.append(single)
    #print(w)
    #print(co)
    mu=np.array([[0,3],[4,-2],[8,3.75]])
    #print(len(mu[0]))
    #print(mu)
    #rv = multivariate_normal.pdf(plot[0],mean=mu[0],cov=co)
    #rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    #print(rv)
    #print(rv)
    
    count=0
    check=0
    for i in range(data_length):
        val=0
        for j in range(len(w)):
            rv = multivariate_normal.pdf(plot[i],mean=mu[j],cov=co[j])
            val=val+rv*w[j]
            #print(val)
        va=np.log(val)
        check=check+va
    
    while 1:
        
        count=count+1
        print("Number of iteration = ",count)
        
        P=[]    
        for i in range(data_length):
            p=[]
            summ=0
            for j in range(len(w)):
                rvv = multivariate_normal.pdf(plot[i],mean=mu[j],cov=co[j])
                p.append(rvv*w[j])
                summ=summ+rvv*w[j]
            for j in range(len(w)):
                p[j]=p[j]/summ
                
            P.append(p)
        #print(P)
        for k in range(len(mu)):
            val=np.zeros((1,2))
            summ1=0
            for i in range(data_length):
                val=val+P[i][k]*plot[i]
                summ1=summ1+P[i][k]
                #print(val)
            
            #for j in range(len(mu[0])):
            mu[k]=val/summ1
                
                
            w[k]=summ1/data_length
        #print(type(mu[0]))
        #print(mu)
        for k in range(len(co)):
            val=np.zeros((2,2))
            summ=0
            for i in range(data_length):
                summ=summ+P[i][k]
                t=plot[i]-mu[k]
                #print(t)
                temp=np.empty((0,2))
                temp=np.vstack([temp,t])
                tt=np.transpose(temp)
                #print()
                #print(tt)
                ttt=np.matmul(tt,temp)
                val=val+P[i][k]*ttt
            co[k]=val/summ
         
        #print(type(co[0]))   
        lnp=0
        for i in range(data_length):
            val=0
            for j in range(len(w)):
                rv = multivariate_normal.pdf(plot[i],mean=mu[j],cov=co[j])
                val=val+rv*w[j]
            #print(val)
            va=np.log(val)
            lnp=lnp+va
        print("Likelihood = ",lnp) 
        
        if abs(lnp-check)<0.01:
            break
        else:
            check=lnp
            
        
        
    #print(count)
    #print(mu)
    for i in range(data_length):
        num1=multivariate_normal.pdf(plot[i],mean=mu[0],cov=co[0])
        num2=multivariate_normal.pdf(plot[i],mean=mu[1],cov=co[1])
        num3=multivariate_normal.pdf(plot[i],mean=mu[2],cov=co[2])
        
        largest=max_num(num1,num2,num3)
        
        if largest==num1:
            plt.scatter(plot[i][0],plot[i][1],marker='o',linewidths=3,color='red')
        elif largest==num2:
            plt.scatter(plot[i][0],plot[i][1],marker='o',linewidths=3,color='green')
        else:
            plt.scatter(plot[i][0],plot[i][1],marker='o',linewidths=3,color='blue')
            
            
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()
    print(len(P[0]))    
def main():
    data=Read_data()
     
     
     
     
if __name__ == "__main__":
    main()