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

# 决策树回归-------------------------------------------------
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

# 决策树分类---------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

iris = myML.DataPre.load_datasets("iris")
# 根据数据的特点，需要设置分层采样
X_train,X_test,Y_train,Y_test = myML.DataPre.train_test_split(iris.data,iris.target,test_size=0.25, random_state=0, stratify=iris.target)

# DecisionTreeClassifier 的用法
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
myML.TreeModel.showModelTest(clf,X_train,Y_train)
myML.TreeModel.showModelTest(clf,X_test,Y_test)

# 测试 DecisionTreeClassifier 的预测性能随 criterion 参数的影响
criterions=['gini','entropy']
for criterion in criterions:
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(X_train, Y_train)
    print("criterion:%s"%criterion)
    print("Training score:%f"%(clf.score(X_train,Y_train)))
    print("Testing score:%f"%(clf.score(X_test,Y_test)))

# 测试 DecisionTreeClassifier 的预测性能随划分类型的影响
splitters=['best','random']
for splitter in splitters:
    clf = DecisionTreeClassifier(splitter=splitter)
    clf.fit(X_train, Y_train)
    print("splitter:%s"%splitter)
    print("Training score:%f"%(clf.score(X_train,Y_train)))
    print("Testing score:%f"%(clf.score(X_test,Y_test)))

# 测试 DecisionTreeClassifier 的预测性能随 max_depth 参数的影响
depths=np.arange(1,20)
training_scores=[]
testing_scores=[]
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, Y_train)
    training_scores.append(clf.score(X_train,Y_train))
    testing_scores.append(clf.score(X_test,Y_test))
## 绘图
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(depths,training_scores,label="traing score",marker='o')
ax.plot(depths,testing_scores,label="testing score",marker='*')
ax.set_xlabel("maxdepth")
ax.set_ylabel("score")
ax.set_title("Decision Tree Classification")
ax.legend(framealpha=0.5,loc='best')
plt.show()


from sklearn.datasets import load_iris
from sklearn import tree
X, y = load_iris(return_X_y=True)
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf.fit(X, y))

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph

