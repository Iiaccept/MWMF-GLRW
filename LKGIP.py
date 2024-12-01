import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy


lncRNA_miRNA_M = np.loadtxt(r"interaction.txt", dtype=int)



#计算lncRNA相似性
def Gaussian():
    row=780
    a = 2

    lncRNA1=np.matlib.zeros((row,row))

    for i in range(0,row):
        for j in range(0,row):
            lncRNA1[i,j]=math.exp(-(1/a)*np.linalg.norm((lncRNA_miRNA_M[i,]-lncRNA_miRNA_M[j,])))


    GlncRNA = lncRNA1
    return GlncRNA



#计算miRNA相似性
def Gaussian1():
    column = 275
    a = 2

    miRNA1=np.matlib.zeros((column,column))

    for i in range(0,column):
        for j in range(0,column):
            miRNA1[i,j]=math.exp(-(1/a)*np.linalg.norm((lncRNA_miRNA_M[:,i]-lncRNA_miRNA_M[:,j])))


    GmiRNA = miRNA1
    return GmiRNA



def main():
    GlncRNA=Gaussian()
    GmiRNA=Gaussian1()

    np.savetxt(r'LKGIP_lncRNA.txt', GlncRNA, delimiter='\t', fmt='%.9f')
    np.savetxt(r'LKGIP_miRNA.txt',  GmiRNA, delimiter='\t', fmt='%.9f')


if __name__ == "__main__":

        main()