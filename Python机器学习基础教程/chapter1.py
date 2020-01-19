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
__mypath__ = MyPath.MyClass_Path("\\Python机器学习基础教程")  # 路径类
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


# ----稀疏矩阵
from scipy import sparse
eye = np.eye(4)
print("NumPy array:\n", eye)
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n", eye_coo)

# ---sklearn
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Keys of iris_dataset:\n", iris_dataset.keys())

#%%

print(iris_dataset['DESCR'][:193] + "\n...")

#%%

print("Target names:", iris_dataset['target_names'])

#%%

print("Feature names:\n", iris_dataset['feature_names'])

#%%

print("Type of data:", type(iris_dataset['data']))

#%%

print("Shape of data:", iris_dataset['data'].shape)

#%%

print("First five rows of data:\n", iris_dataset['data'][:5])

#%%

print("Type of target:", type(iris_dataset['target']))

#%%

print("Shape of target:", iris_dataset['target'].shape)

#%%

print("Target:\n", iris_dataset['target'])

#%% md

#### Measuring Success: Training and Testing Data

#%%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

#%%

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

#%%

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#%% md

#### First Things First: Look at Your Data

#%%
from .bookcode import mglearn
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)

#%% md

#### Building Your First Model: k-Nearest Neighbors

#%%

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

#%%

knn.fit(X_train, y_train)

#%% md

#### Making Predictions

#%%

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

#%%

prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])

#%% md

#### Evaluating the Model

#%%

y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)

#%%

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#%%

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

#%% md

### Summary and Outlook

#%%

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

