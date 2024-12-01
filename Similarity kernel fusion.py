#************************双向均匀归一化预处理****************************
import numpy as np

K1 = 30
K2 = 30


# 从txt文件中读取数据
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            row = [float(x) for x in line.split()]
            data.append(row)
    return np.array(data)

# 列归一化
def column_normalize(matrix):
    normalized_matrix = np.zeros(matrix.shape)
    for j in range(matrix.shape[1]):
        column_sum = np.sum(matrix[:, j])
        normalized_matrix[:, j] = matrix[:, j] / column_sum
    return normalized_matrix


# 计算邻居集合N(包含自身)
def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)

    return N


# 行归一化(邻居归一化）
def row_normalization(S, N):

    result = np.zeros(S.shape)
    for i in range(S.shape[0]):
        num = 0
        denominator=np.sum(S[i, N[i]]) # 分母计算的是邻居的和


       # denominator= np.sum(S[i])   #分母计算的是一整行
        for j in range(S.shape[1]):
            if j in N[i]:
                if denominator != 0:
                    num = num + 1

                    result[i, j] =  S[i, j] / denominator
                else:
                    result[i, j] = 0  # Set to 0 or handle differently based on your requirements
            else:
                result[i, j] = 0
    return result



GKGIP_lncRNA = np.loadtxt('GKGIP_lncRNA.txt')
GKGIP_miRNA = np.loadtxt('GKGIP_miRNA.txt')


LKGIP_lncRNA = np.loadtxt('LKGIP_lncRNA.txt')
LKGIP_miRNA = np.loadtxt('LKGIP_miRNA.txt')



lty = np.loadtxt('interaction.txt')
# 计算邻居集合N
N1 = calculate_neighbors(GKGIP_lncRNA, K1)
N2 = calculate_neighbors(GKGIP_miRNA, K2)
N3 = calculate_neighbors(LKGIP_lncRNA, K1)
N4 = calculate_neighbors(LKGIP_miRNA, K2)
# 执行列归一化
GKGIP_lncRNA_col = column_normalize(GKGIP_lncRNA)
GKGIP_miRNA_col = column_normalize(GKGIP_miRNA)
LKGIP_lncRNA_col = column_normalize(LKGIP_lncRNA)
LKGIP_miRNA_col = column_normalize(LKGIP_miRNA)
# 执行行归一化
GKGIP_lncRNA_row = row_normalization(GKGIP_lncRNA, N1)
GKGIP_miRNA_row = row_normalization(GKGIP_miRNA, N2)
LKGIP_lncRNA_row = row_normalization(LKGIP_lncRNA, N3)
LKGIP_miRNA_row = row_normalization(LKGIP_miRNA, N4)
#第二步
lncRNA_P1=GKGIP_lncRNA_col
lncRNA_P2=LKGIP_lncRNA_col
lncRNA_S1=GKGIP_lncRNA_row
lncRNA_S2=LKGIP_lncRNA_row
alpha_1 =0.01
miRNA_P1=GKGIP_miRNA_col
miRNA_P2=LKGIP_miRNA_col
miRNA_S1=GKGIP_miRNA_row
miRNA_S2=LKGIP_miRNA_row
lncRNA_P2_t=lncRNA_P2
lncRNA_P1_t=lncRNA_P1
for i in range(1000):
    lncRNA_p1=alpha_1*(lncRNA_S1@(lncRNA_P2_t/2)@lncRNA_S1.T)+(1-alpha_1)*(lncRNA_P2/2)
    lncRNA_p2=alpha_1*(lncRNA_S2@(lncRNA_P1_t/2)@lncRNA_S2.T)+(1-alpha_1)*(lncRNA_P1/2)
    err1 = np.sum(np.square(lncRNA_p1-lncRNA_P1_t))
    err2= np.sum(np.square(lncRNA_p2-lncRNA_P2_t))
    if (err1 < 1e-6) and (err2 < 1e-6):
        print("lncRNA迭代的次数：",i)
        break
    lncRNA_P2_t=lncRNA_p2
    lncRNA_P1_t=lncRNA_p1
# #简单平均
lncRNA_sl=0.5*lncRNA_p1+0.5*lncRNA_p2


#*******************************************************************************************
miRNA_P2_t=miRNA_P2
miRNA_P1_t=miRNA_P1
for j in range(1000):
    miRNA_p1=alpha_1*(miRNA_S1@(miRNA_P2_t/2)@miRNA_S1.T)+(1-alpha_1)*(miRNA_P2/2)
    miRNA_p2=alpha_1*(miRNA_S2@(miRNA_P1_t/2)@miRNA_S2.T)+(1-alpha_1)*(miRNA_P1/2)
    err1 = np.sum(np.square(miRNA_p1-miRNA_P1_t))
    err2= np.sum(np.square(miRNA_p2-miRNA_P2_t))
    if (err1 < 1e-6) and (err2 < 1e-6):
        print("miRNA迭代的次数：", i)
        break
    miRNA_P2_t=miRNA_p2
    miRNA_P1_t=miRNA_p1
miRNA_sl=0.5*miRNA_p1+0.5*miRNA_p2


#--------------------------------------------------------------------
def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)
    return N
def compute_weighted_matrix(S1, k1):
    # 计算邻居集合 N_j 和 N_i
    N_i = calculate_neighbors(S1, k1)  # 行的邻居集合
    N_j = calculate_neighbors(S1.T, k1)  # 列的邻居集合
    # 生成 w 矩阵
    w = np.zeros((len(S1), len(S1)))

    for i in range(len(S1)):
        for j in range(len(S1)):
            if i in N_j[j] and j in N_i[i]:
                w[i][j] = 1
            elif i not in N_j[j] and j not in N_i[i]:
                w[i][j] = 0
            else:
                w[i][j] = 0.5
    return w

w1 = compute_weighted_matrix(lncRNA_sl, 30)
w2 = compute_weighted_matrix(miRNA_sl, 30)


average_lncRNA = w1 @ lncRNA_sl
average_miRNA = w2 @ miRNA_sl
np.savetxt('Integration_lncRNA.txt',average_lncRNA,fmt='%6f',delimiter='\t')
np.savetxt('Integration_miRNA.txt',average_miRNA,fmt='%6f',delimiter='\t')


