7-1 求随机矩阵的特征值和特征向量
import numpy as np
a = input().split(" ")
if len(a) != 25:
    print("输入有错！")
else:
    try:
        a = [int(i) for i in a]
        a = [a[i:i + 5] for i in range(0, len(a), 5)]
    except:
        print("输入有错！")
    eig_vals, eig_vectors = np.linalg.eig(a)
    print(eig_vals, '\n', eig_vectors)


7-2 求相关系数矩阵
    import numpy as np
    s1 = input()
    s2 = input()
    n = []
    num = [float(n) for n in s1.split()]
    for i in range(0, len(num), 31):
        x = num[i:i+31]
        n.append(x)
    n = np.matrix(n)
    nn = np.corrcoef(n)
    print(nn)


7-1 线性回归
import numpy as np
from sklearn.linear_model import LinearRegression
a = input()
b = input()
c = input()
X1 = list(map(int, a.split(' ')))
y1 = list(map(float, b.split(' ')))
n = int(c)
X = np.mat(X1).reshape(5, 1)
y = np.mat(y1).reshape(5, 1)

7-3 K均值聚类的实现
import numpy as np
from sklearn.cluster import KMeans
a = input()
b = input()
c = input()
t1 = np.array([float(i) for i in a.split(' ')])
t2, t3 = np.array([int(i) for i in b.split(' ')])
t4 = int(c)
n = np.array(t1).reshape(t2, t3)
kmeans = KMeans(n_clusters=t4)
kmeans.fit(n)
m = kmeans.labels_[0]
print("A公司所在类的中心为：{:.2f},{:.2f}。".format(kmeans.cluster_centers_[m, 0], kmeans.cluster_centers_[m,
1]))

7-4 针对变量的系统聚类实现
import numpy as np
from sklearn.cluster import AgglomerativeClustering
temp = np.array([float(i) for i in input().split(' ')])
n_samplesj, n_features = np.array([int(i) for i in input().split(' ')])
X =np.array(temp).reshape(n_samplesj, n_features)
n_clusters = int(input())
hc=AgglomerativeClustering(n_clusters = n_clusters, affinity = 'correlation', linkage = 'complete')
hc.fit(X.T)
hcl=hc.labels_
if hcl[0]==hcl[2]:
 print("香气和酸质属于一类。")
else:
 print("香气和酸质不属于一类。")


7-1 写出贡献率最大的主成分线性方程
import numpy as np
a = input()
b = input()
b1 = np.array([int (i) for i in b.split(',')])
num = np.array([float(i) for i in a.split(',')]).reshape(b1[0],b1[1])
#均值标准化
x=(num-np.mean(num))/np.std(num)
x1 = np.cov(x.T)
#特征值、特征向量
u,v=np.linalg.eig(x1)
#比较特征值，得出最大的特征值
if u[0]>u[1]:
 index=0
else:
 index=1
print('第1主成分={:.5f}*(x1-{:.2f}){:+.5f}*(x2-{:.2f})'.format(v[0][index],
 np.mean(num, axis=0)[0], v[1][index], np.mean(num, axis=0)[1]))
