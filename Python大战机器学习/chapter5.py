# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("\\Python大战机器学习")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
#------------------------------------------------------------

# ---PCA
from sklearn import decomposition
iris = myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X,y = iris.data,iris.target

# 测试 PCA 的用法 (注意：此PCA基于scipy.linalg来实现SVD分解，因此不能应用于实数矩阵，并且无法适用于超大规模数据。)
pca=decomposition.PCA(n_components=None) # 使用默认的 n_components
pca.fit(X)
print('explained variance ratio : %s'% str(pca.explained_variance_ratio_))

# 绘制经过 PCA 降维到二维之后的样本点
pca=decomposition.PCA(n_components=2) # 目标维度为2维
pca.fit(X) # 训练模型
X_r=pca.transform(X) # 执行降维运算，原始数据集转换到二维
# 绘制二维数据(4维降到2维绘制散点图)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
for label in np.unique(y):
    position = y==label
    ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label)
ax.set_xlabel("X[0]")
ax.set_ylabel("Y[0]")
ax.legend(loc="best")
ax.set_title("PCA")
plt.show()

# 超大规模数据降维 IncrementalPCA
pca=decomposition.IncrementalPCA(n_components=None) # 使用默认的 n_components
pca.fit(X)
print('explained variance ratio : %s'% str(pca.explained_variance_ratio_))


# ---核化线性降维
from sklearn import decomposition
iris = myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X,y = iris.data,iris.target

# 测试 KernelPCA 的用法
kernels=['linear','poly','rbf']
for kernel in kernels:
    kpca=decomposition.KernelPCA(n_components=None,kernel=kernel) # 依次测试四种核函数
    kpca.fit(X)
    print('kernel=%s --> lambdas: %s'% (kernel,kpca.lambdas_))

# 绘制经过 KernelPCA 降维到二维之后的样本点
kernels=['linear','poly','rbf','sigmoid']
fig=plt.figure()
for i,kernel in enumerate(kernels):
    kpca=decomposition.KernelPCA(n_components=2,kernel=kernel)
    kpca.fit(X)
    X_r=kpca.transform(X)# 原始数据集转换到二维
    ax=fig.add_subplot(2,2,i+1) ## 两行两列，每个单元显示一种核函数的 KernelPCA 的效果图
    for label in  np.unique(y):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title("kernel=%s"%kernel)
plt.suptitle("KPCA")
plt.show()

# 绘制经过 使用 poly 核的KernelPCA 降维到二维之后的样本点
fig=plt.figure()
# 每个元素是个元组，代表一组参数（依次为：p 值， gamma 值， r 值）
# p 取值为：3，10
# gamma 取值为 ：1，10
# r 取值为：1，10
# 排列组合一共 8 种组合
import itertools
p=[3,10]; gamma = [1,10] ; r = [1,10]
Params = list(itertools.product(p,gamma,r))# poly 核的参数组成的列表。

for i,(p,gamma,r) in enumerate(Params):
    kpca=decomposition.KernelPCA(n_components=2,kernel='poly'
    ,gamma=gamma,degree=p,coef0=r)  # poly 核，目标为2维
    kpca.fit(X)
    X_r=kpca.transform(X)# 原始数据集转换到二维
    ax=fig.add_subplot(2,4,i+1)## 两行四列，每个单元显示核函数为 poly 的 KernelPCA 一组参数的效果图
    for label in np.unique(y):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label)
    ax.set_xlabel("X[0]")
    ax.set_xticks([]) # 隐藏 x 轴刻度
    ax.set_yticks([]) # 隐藏 y 轴刻度
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title(r"$ (%s (x \cdot z+1)+%s)^{%s}$"%(gamma,r,p))
plt.suptitle("KPCA-Poly")
plt.show()


# 绘制经过 使用 rbf 核的KernelPCA 降维到二维之后的样本点
fig=plt.figure()
Gammas=[0.5,1,4,10]# rbf 核的参数组成的列表。每个参数就是 gamma值
for i,gamma in enumerate(Gammas):
    kpca=decomposition.KernelPCA(n_components=2,kernel='rbf',gamma=gamma)
    kpca.fit(X)
    X_r=kpca.transform(X)# 原始数据集转换到二维
    ax=fig.add_subplot(2,2,i+1)## 两行两列，每个单元显示核函数为 rbf 的 KernelPCA 一组参数的效果图
    for label in  np.unique(y):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label)
    ax.set_xlabel("X[0]")
    ax.set_xticks([]) # 隐藏 x 轴刻度
    ax.set_yticks([]) # 隐藏 y 轴刻度
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title(r"$\exp(-%s||x-z||^2)$"%gamma)
plt.suptitle("KPCA-rbf")
plt.show()

#  绘制经过 使用 sigmoid 核的KernelPCA 降维到二维之后的样本点

fig=plt.figure()
Params=[(0.01,0.1),(0.01,0.2),(0.1,0.1),(0.1,0.2),(0.2,0.1),(0.2,0.2)]# sigmoid 核的参数组成的列表。
    # 每个元素就是一种参数组合（依次为 gamma,coef0）
    # gamma 取值为： 0.01，0.1，0.2
    # coef0 取值为： 0.1,0.2
    # 排列组合一共有 6 种组合
for i,(gamma,r) in enumerate(Params):
    kpca=decomposition.KernelPCA(n_components=2,kernel='sigmoid',gamma=gamma,coef0=r)
    kpca.fit(X)
    X_r=kpca.transform(X)# 原始数据集转换到二维
    ax=fig.add_subplot(3,2,i+1)## 三行两列，每个单元显示核函数为 sigmoid 的 KernelPCA 一组参数的效果图
    for label  in np.unique(y):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label)
    ax.set_xlabel("X[0]")
    ax.set_xticks([]) # 隐藏 x 轴刻度
    ax.set_yticks([]) # 隐藏 y 轴刻度
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title(r"$\tanh(%s(x\cdot z)+%s)$"%(gamma,r))
plt.suptitle("KPCA-sigmoid")
plt.show()




