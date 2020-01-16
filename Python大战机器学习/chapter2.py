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



from sklearn.tree import DecisionTreeRegressor

# --- random data
n=1000
np.random.seed(0)
X = 5 * np.random.rand(n, 1)
y = np.sin(X).ravel()
noise_num=(int)(n/5)
y[::5] += 3 * (0.5 - np.random.rand(noise_num)) # 每第5个样本，就在该样本的值上添加噪音
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(X,y,test_size=0.25,random_state=1)

# 计算决策树
regr = DecisionTreeRegressor()
regr.fit(X_train, y_train)
myML.TreeModel.showModelTest(regr,X_train,y_train)
myML.TreeModel.showModelTest(regr,X_test,y_test)

##绘图
myML.TreeModel.DecisionTree_PlotPredict(regr,X_train,y_train,"train sample",show=False)
myML.TreeModel.DecisionTree_PlotPredict(regr,X_test,y_test,"test sample",show=True)

# 测试 DecisionTreeRegressor 预测性能随划分类型的影响
splitters=['best','random']
for splitter in splitters:
    regr = DecisionTreeRegressor(splitter=splitter)
    regr.fit(X_train, y_train)
    print("Splitter %s"%splitter)
    print("Training score:%f"%(regr.score(X_train,y_train)))
    print("Testing score:%f"%(regr.score(X_test,y_test)))

# 测试 DecisionTreeRegressor 预测性能随  max_depth 的影响
depths=np.arange(1,20)
training_scores=[]
testing_scores=[]
for depth in depths:
    regr = DecisionTreeRegressor(max_depth=depth)
    regr.fit(X_train, y_train)
    training_scores.append(regr.score(X_train,y_train))
    testing_scores.append(regr.score(X_test,y_test))
## 绘图
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(depths,training_scores,label="traing score")
ax.plot(depths,testing_scores,label="testing score")
ax.set_xlabel("maxdepth")
ax.set_ylabel("score")
ax.set_title("Decision Tree Regression")
ax.legend(framealpha=0.5)
plt.show()









