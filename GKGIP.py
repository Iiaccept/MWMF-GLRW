import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy


lncRNA_miRNA_M = np.loadtxt(r"interaction.txt", dtype=int)

print(lncRNA_miRNA_M.shape[1])


#计算lncRNA高斯轮廓核相似性
def Gaussian():
    row=780
    sum=0
    lncRNA1=np.matlib.zeros((row,row))
    for i in range(0,row):
        a=np.linalg.norm(lncRNA_miRNA_M[i,])*np.linalg.norm(lncRNA_miRNA_M[i,])
        sum=sum+a
    ps=row/sum
    for i in range(0,row):
        for j in range(0,row):
            lncRNA1[i,j]=math.exp(-ps*np.linalg.norm(lncRNA_miRNA_M[i,]-lncRNA_miRNA_M[j,])*np.linalg.norm(lncRNA_miRNA_M[i,]-lncRNA_miRNA_M[j,]))


    GlncRNA = lncRNA1
    return GlncRNA
#计算mirna高斯轮廓核相似性
def Gaussian1():
    column=275
    sum=0
    miRNA1=np.matlib.zeros((column,column))
    for i in range(0,column):
        a=np.linalg.norm(lncRNA_miRNA_M[:,i])*np.linalg.norm(lncRNA_miRNA_M[:,i])
        sum=sum+a
    ps=column/sum
    for i in range(0,column):
        for j in range(0,column):
            miRNA1[i,j]=math.exp(-ps*np.linalg.norm(lncRNA_miRNA_M[:,i]-lncRNA_miRNA_M[:,j])*np.linalg.norm(lncRNA_miRNA_M[:,i]-lncRNA_miRNA_M[:,j]))


    GmiRNA = miRNA1
    return GmiRNA

def main():
    GKSS=Gaussian()
    GKmiRNA=Gaussian1()
    np.savetxt(r'GKGIP_lncRNA.txt', GKSS, delimiter='\t', fmt='%.9f')
    np.savetxt(r'GKGIP_miRNA.txt',  GKmiRNA, delimiter='\t', fmt='%.9f')

if __name__ == "__main__":

        main()