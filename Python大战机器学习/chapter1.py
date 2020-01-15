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
myML = MyMachineLearning.MyClass_MachineLearning() # 机器学习综合类
#------------------------------------------------------------

"""
    广义线性模型
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    LinearRegression
    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
data = myML.DataPre.load_datasets(mode="diabetes")

X_train,X_test,Y_train,Y_test = myML.DataPre.train_test_split(data.data,data.target,test_size=0.25,random_state=0)

from sklearn import  linear_model

# ---LinearRegression线性回归
regr = linear_model.LinearRegression().fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)

# ---岭回归
regr = linear_model.Ridge().fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)
# 测试alpha
alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
scores=[]
for i,alpha in enumerate(alphas):
    regr = linear_model.Ridge(alpha=alpha).fit(X_test, Y_test)
    scores.append(regr.score(X_test, Y_test))
## 绘图
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(alphas,scores)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"score")
ax.set_xscale('log')
ax.set_title("Ridge")
plt.show()



