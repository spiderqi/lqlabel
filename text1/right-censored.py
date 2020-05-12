# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:00:20 2020

@author: 15272
"""

import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import numpy as np

df=pd.read_excel('D:/生存分析/text1/sur.xlsx')#load data

count1=collections.Counter(df['time'])
count2=collections.Counter(df['time'])
n=len(df['time1'])#total label
for i in range(n):
    if df['status'][i]==False:
        count2[df['time'][i]]-=1

def surer(time,count1,count2,y=21,st=1,std=0,st2=1):
    surname=[[0,0,21,1,0,0]]
    x=list(set(time))#setting 
    x.sort()
    for i in x:
        st=st*(1-count2[i]/y)
        if count2[i] !=0:
            std=std+(count2[i]/(y*(y-count2[i])))
            st2=st**2*std
            surname.append([i,count2[i],y,st,std,st2])
        y=y-count1[i]
    surname=pd.DataFrame(surname,columns=['time','events','all risk','St','std','St2'])
    return surname

def Ht(time,count1,count2,y=21,ht=0,vht=0):
    surname=[[0,0,21,0,0]]
    x=list(set(time))#setting 
    x.sort()
    for i in x:
        if count2[i]!=0:
            ht+=count2[i]/y
            vht+=count2[i]/(y**2)
            surname.append([i,count2[i],y,ht,vht])
        y-=count1[i]
    surname=pd.DataFrame(surname,columns=['time','events','all risk','Ht','Ht2'])
    return surname
sur2=surer(df['time'],count1,count2)

sur=Ht(df['time'],count1,count2)
Ht=sur2['St'].apply(lambda x: -log(x))
