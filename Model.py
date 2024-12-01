# -*- codeing = utf-8 -*-
# @Time : 2023/3/24 21:17
# @Author : 刘体耀
# @File : TSPN-CMF.py
# @Software: PyCharm
import array

import numpy as np
#import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import copy
import fnorm
import Weight



lncRNA1 = np.loadtxt(r"GKGIP_lncRNA.txt", dtype=float)
miRNA1 = np.loadtxt(r"GKGIP_miRNA.txt", dtype=float)
lncRNAW1 = Weight.calculate_weight_matrix(lncRNA1, 110)
miRNAW1 = Weight.calculate_weight_matrix(miRNA1, 105)

lncRNA2 = np.loadtxt(r"LKGIP_lncRNA.txt", dtype=float)
miRNA2 = np.loadtxt(r"LKGIP_miRNA.txt", dtype=float)
lncRNAW2 = Weight.calculate_weight_matrix(lncRNA1, 100)
miRNAW2 = Weight.calculate_weight_matrix(miRNA1, 100)




lncRNA = np.loadtxt(r"Integration_lncRNA.txt", dtype=float)
miRNA = np.loadtxt(r"Integration_miRNA.txt", dtype=float)


Y1= np.loadtxt(r"interaction.txt",dtype=float)
lncRNA_miRNA_k = np.loadtxt(r"known.txt",dtype=int)
lncRNA_miRNA_uk = np.loadtxt(r"unknown.txt",dtype=int)




normWrr = fnorm.fNorm(lncRNA)
normWdd = fnorm.fNorm(miRNA)




def run_MC(A):
    alpha0 = 0.9
    alpha = 0.5

    l = 1
    r = 2

    R0 = A / np.sum(A)
    Rt = R0.copy()



    for t in range(1, max(l, r) + 1):
        ftl = 0
        ftr = 0
        Rt =  (1 - alpha0) * normWrr @ Rt @ normWdd + alpha0 * R0
        if t <= l:
            nRtleft =  (1 - alpha) * normWrr @ Rt + alpha * R0

            ftl = 1
        if t <= r:
            nRtright =   (1 - alpha) * Rt @ normWdd + alpha * R0


            ftr = 1

        Rt = (ftl * nRtleft + ftr * nRtright) / (ftl + ftr)


    return Rt



#多视角加权矩阵分解
def TCMF(alpha, beta,gamma,Y, maxiter,A,B,C,lncRNA1,miRNA1,lncRNAW1,miRNAW1,lncRNA2,miRNA2,lncRNAW2,miRNAW2):

    iter0=1
    while True:

        a = np.dot(Y,B)+ beta * (np.dot(np.multiply(lncRNAW1,lncRNA1),A) + np.dot(np.multiply(lncRNAW2,lncRNA2),A)  )
        b = np.dot(np.transpose(B),B)+alpha*C+beta*np.dot(np.transpose(A),A)
        A = np.dot(a, np.linalg.inv(b))
        c = np.dot(np.transpose(Y),A)+ gamma * (np.dot(np.multiply(miRNAW1,miRNA1),B) + np.dot(np.multiply(miRNAW2,miRNA2),B))
        d = np.dot(np.transpose(A), A) + alpha * C + gamma * np.dot(np.transpose(B), B)

        B = np.dot(c, np.linalg.inv(d))



        if iter0 >= maxiter:


            break
        iter0 = iter0 + 1
    Y= np.dot(A,np.transpose(B))
    Y_recover = Y
    return Y_recover


def run_MC_2(Y):
    maxiter = 100
    alpha = 2
    beta = 1
    gamma = 1

    #SVD

    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r = 30
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]

    V= V[0:r,:]

    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    C = np.zeros([r,r])
    for i in range(0,r):
        C[i][i] = 1


    Y = TCMF(alpha, beta,gamma,Y, maxiter,A,B,C,lncRNA1,miRNA1,lncRNAW1,miRNAW1,lncRNA2,miRNA2,lncRNAW2,miRNAW2)
    Smmi = Y
    return Smmi



def main():

    roc_sum, time = 0, 0

    kf = KFold(n_splits=5, shuffle=True,random_state=9999)#
    for train_index,test_index in kf.split(lncRNA_miRNA_k):
        X_2 = copy.deepcopy(Y1)

        for index in test_index:
            X_2[lncRNA_miRNA_k[index, 0] , lncRNA_miRNA_k[index, 1] ] = 0

        lty = run_MC_2(X_2)

        M_1 = run_MC(lty)



        Label = np.zeros(lncRNA_miRNA_uk.shape[0] + test_index.size)
        Score = np.zeros(lncRNA_miRNA_uk.shape[0] + test_index.size)
        i , j = 0 , 0
        for s_index in test_index:
            Label[i] = 1
            Score[i] = M_1[lncRNA_miRNA_k[s_index,0],lncRNA_miRNA_k[s_index,1]]

            i = i + 1
        for i in range(test_index.size, lncRNA_miRNA_uk.shape[0] + test_index.size):
            Score[i] = M_1[lncRNA_miRNA_uk[j,0],lncRNA_miRNA_uk[j,1]]
            j = j + 1
        fpr, tpr, thersholds = roc_curve(y_true=Label, y_score=Score, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        roc_sum = roc_sum + roc_auc
        time += 1
        s=roc_sum/time
        print(time,roc_auc,roc_sum,s)
        while(time==5):
            return s


if __name__ == "__main__":
    total,time=0,0
    for i in range(0,1):
        l=main()
        print("\n")
        total=total+l
        time +=1
        print(time,total/time)
        print("\n")
