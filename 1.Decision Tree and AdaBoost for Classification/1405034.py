# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:33:31 2018

@author: Raju
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import binarize
eps=np.finfo(float).eps



tracklist=[]

def Mean_Calculation(dataset,column):
    #result1=dataset.sort_values(by=[column])
    length=len(dataset)
    sum=0
    for val in dataset[column]:
        sum=sum+val
    mean=sum/length
    
    return mean
    

def Pre_Processing_Telco():
    data=pd.read_csv('Telco.csv')
    data_length=len(data)
    #print(data.iloc[0:4])
    #print(data_length)
    data1=data
    mean1=Mean_Calculation(dataset=data1,column="tenure")
    #print(mean1)
    #data1=binarize(data["tenure"].values.reshape(0,1),mean)
    data1["tenure"]= (data1["tenure"] >= mean1).astype(int)
    
    data2=data1
    mean2=Mean_Calculation(dataset=data2,column="MonthlyCharges")
    #print(mean2)
    data2["MonthlyCharges"]= (data2["MonthlyCharges"] >= mean2).astype(int)
    
    #data3["TotalCharges"]= (data3["TotalCharges"]).astype(float)
    #data3["TotalCharges"].apply(lambda x:float(x))
    #print(type(data3["TotalCharges"]))
    data2=data2.drop("TotalCharges",axis=1)
    data2=data2.drop("customerID",axis=1)
    data2["Churn"]= (data2["Churn"] == "Yes").astype(int)
    
    split = 0.8*len(data2)
    #print(split)
    split=int(split)
    data_train=data2.iloc[0:split]
    data_test=data2.iloc[(split+1):data_length]
    #print(len(data_train)," ",len(data_test))
    #print(split)
    """

    data3.TotalCharges = data3.TotalCharges.astype(float)
    mean3=Mean_Calculation(dataset=data3,column="TotalCharges")
    print(mean3)
    data3["TotalCharges"]= (data3["TotalCharges"] >= mean3).astype(int)
    """
    #print(data2.isnull().sum())
    data_test=data_test.reset_index(drop=True)
    return data_train,data_test


def Pre_Processing_Adult():
    columns=["age","workclass","fnlwgt","education","education-num","marital-status",
             "occupation","relationship","race","sex","capital-gain","capital-loss",
             "hours-per-week","native-country","class"]
    
    data=pd.read_csv('adult.csv',names=columns)
    data_length=len(data)
    #print(data_length)
    data1=data
    #print(data1["class"][0])
    data1["class"]= (data1["class"] != " >50K").astype(int)
    
    mean1=Mean_Calculation(dataset=data1,column="age")
    data1["age"]= (data1["age"] >= mean1).astype(int)
    
    data2=data1
    mean2=Mean_Calculation(dataset=data2,column="fnlwgt")
    data2["fnlwgt"]= (data2["fnlwgt"] >= mean2).astype(int)
    
    data3=data2
    mean3=Mean_Calculation(dataset=data3,column="capital-gain")
    data3["capital-gain"]= (data3["capital-gain"] >= mean3).astype(int)
    
    data4=data3
    mean4=Mean_Calculation(dataset=data4,column="capital-loss")
    data4["capital-loss"]= (data4["capital-loss"] >= mean4).astype(int)
    
    data5=data4
    mean5=Mean_Calculation(dataset=data5,column="hours-per-week")
    data5["hours-per-week"]= (data5["hours-per-week"] >= mean5).astype(int)
    
    data6=data5
    mean6=Mean_Calculation(dataset=data6,column="education-num")
    data6["education-num"]= (data6["education-num"] >= mean6).astype(int)
    data6=data6.drop("native-country",axis=1)
    data6=data6.drop("occupation",axis=1)
    
    #print(data6.isnull().sum())
    data7=data6
    maxx=data7["workclass"].value_counts().idxmax()
    #print(maxx)
    """
    mask = data7.workclass = "?"
    column_name = "workclass"
    data7.loc[mask, column_name] = maxx
    """
    
    data7.loc[data7.workclass == " ?", "workclass"] = maxx
    
    split = 0.8*len(data7)
    #print(split)
    split=int(split)
    data_train=data7.iloc[0:split]
    data_test=data7.iloc[(split+1):data_length]
    data_test=data_test.reset_index(drop=True)
    return data_train,data_test
    
def Pre_Processing_Fraud():
    data=pd.read_csv('fraud.csv')
    data_length=len(data)
    #print(data.iloc[0:4])
    #print(data_length)
    #print(data.isnull().sum())
    
    data1=data[data["Class"]==1]
    #print(len(data1))
    
    data0=data[data["Class"]==0]
    #print(len(data0))
    
    data2=data1.append(data0.iloc[0:20000])
    #print(len(data2))
    
    data3=data2
    
    mean=Mean_Calculation(dataset=data3,column="Time")
    data3["Time"]= (data3["Time"] >= mean).astype(int)
    
    mean1=Mean_Calculation(dataset=data3,column="V1")
    data3["V1"]= (data3["V1"] >= mean1).astype(int)
    
    mean2=Mean_Calculation(dataset=data3,column="V2")
    data3["V2"]= (data3["V2"] >= mean2).astype(int)
    
    mean3=Mean_Calculation(dataset=data3,column="V3")
    data3["V3"]= (data3["V3"] >= mean3).astype(int)
    
    mean4=Mean_Calculation(dataset=data3,column="V4")
    data3["V4"]= (data3["V4"] >= mean4).astype(int)
    
    mean5=Mean_Calculation(dataset=data3,column="V5")
    data3["V5"]= (data3["V5"] >= mean5).astype(int)
    
    mean6=Mean_Calculation(dataset=data3,column="V6")
    data3["V6"]= (data3["V6"] >= mean6).astype(int)
    
    mean7=Mean_Calculation(dataset=data3,column="V7")
    data3["V7"]= (data3["V7"] >= mean7).astype(int)
    
    mean8=Mean_Calculation(dataset=data3,column="V8")
    data3["V8"]= (data3["V8"] >= mean8).astype(int)
    
    mean9=Mean_Calculation(dataset=data3,column="V9")
    data3["V9"]= (data3["V9"] >= mean9).astype(int)
    
    mean10=Mean_Calculation(dataset=data3,column="V10")
    data3["V10"]= (data3["V10"] >= mean10).astype(int)
    
    mean11=Mean_Calculation(dataset=data3,column="V11")
    data3["V11"]= (data3["V11"] >= mean11).astype(int)
    
    mean12=Mean_Calculation(dataset=data3,column="V12")
    data3["V12"]= (data3["V12"] >= mean12).astype(int)
    
    mean13=Mean_Calculation(dataset=data3,column="V13")
    data3["V13"]= (data3["V13"] >= mean13).astype(int)
    
    mean14=Mean_Calculation(dataset=data3,column="V14")
    data3["V14"]= (data3["V14"] >= mean14).astype(int)
    
    mean15=Mean_Calculation(dataset=data3,column="V15")
    data3["V15"]= (data3["V15"] >= mean15).astype(int)
    
    mean16=Mean_Calculation(dataset=data3,column="V16")
    data3["V16"]= (data3["V16"] >= mean16).astype(int)
    
    mean17=Mean_Calculation(dataset=data3,column="V17")
    data3["V17"]= (data3["V17"] >= mean17).astype(int)
    
    mean18=Mean_Calculation(dataset=data3,column="V18")
    data3["V18"]= (data3["V18"] >= mean18).astype(int)
    
    mean19=Mean_Calculation(dataset=data3,column="V19")
    data3["V19"]= (data3["V19"] >= mean19).astype(int)
    
    mean20=Mean_Calculation(dataset=data3,column="V20")
    data3["V20"]= (data3["V20"] >= mean20).astype(int)
    
    mean21=Mean_Calculation(dataset=data3,column="V21")
    data3["V21"]= (data3["V21"] >= mean21).astype(int)
    
    mean22=Mean_Calculation(dataset=data3,column="V22")
    data3["V22"]= (data3["V22"] >= mean22).astype(int)
    
    mean23=Mean_Calculation(dataset=data3,column="V23")
    data3["V23"]= (data3["V23"] >= mean23).astype(int)
    
    mean24=Mean_Calculation(dataset=data3,column="V24")
    data3["V24"]= (data3["V24"] >= mean24).astype(int)
    
    mean25=Mean_Calculation(dataset=data3,column="V25")
    data3["V25"]= (data3["V25"] >= mean25).astype(int)
    
    mean26=Mean_Calculation(dataset=data3,column="V26")
    data3["V26"]= (data3["V26"] >= mean26).astype(int)
    
    mean27=Mean_Calculation(dataset=data3,column="V27")
    data3["V27"]= (data3["V27"] >= mean27).astype(int)
    
    mean28=Mean_Calculation(dataset=data3,column="V28")
    data3["V28"]= (data3["V28"] >= mean28).astype(int)
    
    meanam=Mean_Calculation(dataset=data3,column="Amount")
    data3["Amount"]= (data3["Amount"] >= meanam).astype(int)
    
    split = 0.8*len(data3)
    #print(split)
    split=int(split)
    data_train=data3.iloc[0:split]
    data_test=data3.iloc[(split+1):data_length]
    
    data_train=data_train.reset_index(drop=True)
    data_test=data_test.reset_index(drop=True)
    return data_train,data_test



def Entropy(dataset,column):
    column_dis=set(dataset[column])
    entropy=0
    for value in column_dis:
        #count =len(dataset[dataset[column]==value])
        count=0
        for val in dataset[column]:
            if val==value:
                count=count+1
        prob=count/len(dataset)
        entropy=entropy-prob*math.log(prob,2)
        
    return entropy


def Information_gain(dataset,attribute,parent_entropy,column):
    attribute_dis=set(dataset[attribute])
    child_entropy=0
    for value in attribute_dis:
        #backup=pd.DataFrame(dataset)
        count=0
        #prob=Entropy(dataset=dataset[dataset[attribute]==value])
        for val in dataset[attribute]:
            if val==value:
                count=count+1
        portion=count/len(dataset)
        #backup[backup.attribute==value]
        ent=Entropy(dataset=dataset[dataset[attribute]==value],column=column)
        child_entropy=child_entropy+ent*portion
        
    info_gain=parent_entropy-child_entropy
    return info_gain

def Plurality_Value(dataset,column):
    column_dis=set(dataset[column])
    max_count=0
    for value in column_dis:
        #count =len(dataset[dataset[column]==value])
        count=0
        for val in dataset[column]:
            if val==value:
                count=count+1
        if count>max_count:
            max_count=count
            att=value
    return att

def Check_all_examples_same(dataset,column):
    val=dataset.iloc[0][column]
    count=0
    for c in dataset[column]:
        if c==val:
            count=count+1
    if count==len(dataset):
        return 1,val
    else :
        return 0,val
    
    
class Tree_node:
  def __init__(self):
    self.isLeaf=False
    self.Child= {}
    #self.Outcome={}
    self.Attribute=None
    self.depth=0
    self.type=None
    
    
def Importance(dataset,attributes,parent_entropy,column):
    #attributes=dataset.columns.values
    #print("att ",len(attributes))
    max_info=-1
    for a in attributes:
        if a!=column:
            val=Information_gain(dataset=dataset,attribute=a,parent_entropy=parent_entropy,column=column)
            #print(a," ",val)
            if val>max_info:
                max_info=val
                att=a
    
    return att


    
def Decision_Tree_Learning(dataset,attributes,parent,column,depth,reqdepth):
    #print(dataset.iloc[0:14])
    node=Tree_node()
    #print("dep ",depth)
    """
    if parent is None:
        node.depth=0
    else:
        node.depth=parent.depth+1
    """
    
    val,classification=Check_all_examples_same(dataset=dataset,column=column)
    
    if len(dataset)==0:
        node.isLeaf=True
        node.type=Plurality_Value(dataset=parent,column=column)
        #print("exs sesh")
        #print(node)
        return node
    
    elif depth==reqdepth:
        node.isLeaf=True
        node.type=Plurality_Value(dataset=parent,column=column)
        return node
    
    elif val==1:
        node.isLeaf=True
        node.type=classification
        #print(node)
        #print("all same")
        return node
    
    
# =============================================================================
#     elif len(attributes)==1:
#         node.isLeaf=True
#         node.type=Plurality_Value(dataset=dataset,column=column)
#         #print(node)
#         #print("hell 3")
#         return node
# =============================================================================
    
    else:
        entropy=Entropy(dataset=dataset,column=column)
        
        
        attrlist=[]
        for v in dataset.columns.values:
            attrlist.append(v)
            
        for x in tracklist:
            for y in attrlist:
                if x==y:
                    attrlist.remove(y)
        if len(attrlist)==1:
            node.isLeaf=True
            node.type=Plurality_Value(dataset=dataset,column=column)
            #print(node)
            #print("att sesh")
            return node
        A=Importance(dataset=dataset,attributes=attrlist,parent_entropy=entropy,column=column)
        #print("Importance  ",A)
        node.Attribute=A
        tracklist.append(A)
        #print(tracklist)
        
        #node.Outcome=set(dataset[A])
        for value in set(dataset[A]):
            #att=attributes
            #print(value)
            """
            attlist=[]
            for v in dataset.columns.values:
                attlist.append(v)
            
            for x in tracklist:
                for y in attlist:
                    if x==y:
                        attlist.remove(y)
                        
            """
            #attlist.remove(A)
            #print(attlist)
            subtree=dataset[dataset[A]==value]
            #print(subtree)
            node.Child[value]=Decision_Tree_Learning(subtree,attrlist,dataset,column,depth+1,reqdepth)
        
        return node

def check(row,decisionTree):
    
    while decisionTree.isLeaf!=True:
        att=decisionTree.Attribute
        val=row[att]
        decisionTree=decisionTree.Child[val]
        
    return decisionTree.type   
    

def Decision_Tree_check(dataset,decisionTree,column):
    truepos=0
    trueneg=0
    falsepos=0
    falseneg=0
    l=len(dataset)
    #print("yaa")
    for i in range(l):
        #print(dataset[column][i])
        typ=check(row=dataset.iloc[i],decisionTree=decisionTree)
        #print(typ," ",dataset[column][i])
        if typ==1 and dataset[column][i]==1:
            truepos=truepos+1
        elif typ==0 and dataset[column][i]==0:
            trueneg=trueneg+1
        elif typ==0 and dataset[column][i]==1:
            falseneg=falseneg+1
        elif typ==1 and dataset[column][i]==0:
            falsepos=falsepos+1
    
    return truepos,trueneg,falsepos,falseneg


def AdaBoost(dataset,K,column):
    global tracklist
    N=len(dataset)
    w=[1/N]*N
    h=[]*K
    z=[]*K
    data=dataset
    for k in range(K):
        #print(w)
        data=dataset.sample(n=N,replace=True,weights=w)
        #print(data)
        temp=Decision_Tree_Learning(dataset=data,attributes=data.columns.values,parent=None,column=column,depth=0,reqdepth=1)
        tracklist.clear()
        error=0
        for j in range(N):
            x=check(row=dataset.iloc[j],decisionTree=temp)
            #print(x," ",dataset[column][j]," ",j)
            if x!=dataset[column][j]:
                error=error+w[j]
        #print(error)
        
        if error>0.5:
            continue
        for j in range(N):
            xx=check(row=dataset.iloc[j],decisionTree=temp)
            if xx==dataset[column][j]:
                w[j]=w[j]*error/(1-error+eps)
        
        sum=0
        for j in range(len(w)):
            sum=sum+w[j]
            
        #print(sum)
        for j in range(len(w)):
            w[j]=w[j]/(sum+eps)
            
        h.append(temp)
        app=math.log((1-error)/(error+eps),2)
        z.append(app)
        
    return h,z
        

def AdaBoost_check(h,z,dataset,column):
    truepos=0
    trueneg=0
    falsepos=0
    falseneg=0
    l=len(dataset)
    hl=len(h)
    
    for i in range(l):
        sum=0
        for j in range(hl):
            chk=check(row=dataset.iloc[i],decisionTree=h[j])
            if chk==dataset[column][i]:
                w=1
            else:
                w=-1
            sum=sum+w*z[j]
         
        if sum>=0 and dataset[column][i]==1:
            truepos=truepos+1
        elif sum<0 and dataset[column][i]==0:
            trueneg=trueneg+1
        elif sum<0 and dataset[column][i]==1:
            falseneg=falseneg+1
        elif sum>=0 and dataset[column][i]==0:
            falsepos=falsepos+1
            
            
    return truepos,trueneg,falsepos,falseneg
    
def main():
    

    #data_telco_training,data_telco_test=Pre_Processing_Telco()
    #data_adult_training,data_adult_test=Pre_Processing_Adult()
    data_fraud_training,data_fraud_test=Pre_Processing_Fraud()

    
    
    #d=Decision_Tree_Learning(dataset=data_telco_training,attributes=data_telco_training.columns.values,parent=None,column="Churn",depth=0,reqdepth=30)
    #d=Decision_Tree_Learning(dataset=data_adult_training,attributes=data_adult_training.columns.values,parent=None,column="class",depth=0,reqdepth=30)
    #d=Decision_Tree_Learning(dataset=data_fraud_training,attributes=data_fraud_training.columns.values,parent=None,column="Class",depth=0,reqdepth=30)
    
    #Telco
    #tp,tn,fp,fn=Decision_Tree_check(dataset=data_telco_test,decisionTree=d,column="Churn")
    #print(tp," ",tn," ",fp," ",fn)
    
    
    #Adult
    #tp,tn,fp,fn=Decision_Tree_check(dataset=data_adult_test,decisionTree=d,column="class")
    #print(tp," ",tn," ",fp," ",fn)
    
    #Fraud
    #tp,tn,fp,fn=Decision_Tree_check(dataset=data_fraud_training,decisionTree=d,column="Class")
    #print(tp," ",tn," ",fp," ",fn)
    
    
    #h,z=AdaBoost(dataset=data_telco_training,K=20,column="Churn")
    #h,z=AdaBoost(dataset=data_adult_training,K=20,column="class")
    h,z=AdaBoost(dataset=data_fraud_training,K=20,column="Class")
    
    
    #Telco
    #atp,atn,afp,afn=AdaBoost_check(h=h,z=z,dataset=data_telco_test,column="Churn")
    #print(atp," ",atn," ",afp," ",afn)
    
    #Adult
    #atp,atn,afp,afn=AdaBoost_check(h=h,z=z,dataset=data_adult_test,column="class")
    #print(atp," ",atn," ",afp," ",afn)
    
    #Fraud
    atp,atn,afp,afn=AdaBoost_check(h=h,z=z,dataset=data_fraud_test,column="Class")
    print(atp," ",atn," ",afp," ",afn)
    
# =============================================================================
#     accuracy=(tp+tn)/(tp+tn+fp+fn)*100
#     true_positive_rate=tp/(tp+fn)*100
#     true_negative_rate=tn/(tn+fp)*100
#     positive_pred_value=tp/(tp+fp)*100
#     false_dis_rate=fp/(fp+tp)*100
#     f1_score=(2*tp)/(2*tp+fp+fn)*100
#     
#     print(accuracy)
#     print(true_positive_rate)
#     print(true_negative_rate)
#     print(positive_pred_value)
#     print(false_dis_rate)
#     print(f1_score)
# =============================================================================
    accuracy=(atp+atn)/(atp+atn+afp+afn)*100
    print(accuracy)
if __name__ == "__main__":
    main()
