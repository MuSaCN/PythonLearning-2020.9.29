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





