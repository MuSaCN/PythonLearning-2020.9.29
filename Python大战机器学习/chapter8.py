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

# ---感知机模型
# data
train_data0 = myML.Neur.creatTestData(n=20,no_linear=False,value0=10,value1=20,plot=False)
train_data1 = myML.Neur.creatTestData(n=20,no_linear=True,value0=10,value1=20,plot=False)

# ---对线性可分数据集执行感知机的原始算法并绘制分离超平面
data=train_data0 #产生线性可分数据集
w_0= np.ones((3,1),dtype=float) # 初始化 权重
w,b,num = myML.Neur.PerceptronAlgorithm(data,w_0,eta=0.1,b_0=1) # 执行感知机的原始形式
### 绘图
myML.Neur.plotSamples(data,w=w,b=b)


# ---对线性可分数据集执行感知机的原始算法和对偶形式算法，并绘制分离超平面
data=train_data0
## 执行原始形式的算法
w_1,b_1,num_1=myML.Neur.PerceptronAlgorithm(data,w_0=np.ones((3,1),dtype=float),eta=0.1,b_0=1)
myML.Neur.plotSamples(data,w=w_1,b=b_1)
## 执行对偶形式的算法
import time
print(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
w_2,b_2,num_2,alpha=myML.Neur.PerceptronAlgorithm_dual(train_data=data,alpha_0=np.zeros((data.shape[0],1)),eta=0.1,b_0=0)
myML.Neur.plotSamples(data,w=w_2,b=b_2)
print(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
#
print("w_1,b_1",w_1,b_1)
print("w_2,b_2",w_2,b_2)

# 测试学习率对于感知机两种形式算法的收敛速度的影响
data=train_data0 # 线性可分数据集
etas=np.linspace(0.01,1,num=25,endpoint=False)
w_0,b_0,alpha_0=np.ones((3,1)),0,np.zeros((data.shape[0],1))
etas=np.linspace(0.01,1,num=25,endpoint=False)
nums1=[]
for eta in etas:
    _,_,num_1=myML.Neur.PerceptronAlgorithm(data,w_0=w_0,eta=eta,b_0=b_0) # 获取原始形式算法的迭代次数
    nums1.append(num_1)
fig=plt.figure()
fig.suptitle("perceptron")
ax=fig.add_subplot(1,1,1)
ax.set_xlabel(r'$\eta$')
ax.plot(etas,np.array(nums1),label='orignal iteraton times')
ax.legend(loc="best",framealpha=0.5)
plt.show()



# ------多层神经网络
from sklearn import neural_network
train_data=myML.Neur.creatTestData(500,no_linear=True,value0=10,value1=20,datadim=2,plot=True)

# ---使用 MLPClassifier绘制预测结果
train_x=train_data[:,:-1]
train_y=train_data[:,-1]
clf=neural_network.MLPClassifier(activation='logistic',max_iter=1000)# 构造分类器实例
clf.fit(train_x,train_y) # 训练分类器
print(clf.score(train_x,train_y)) # 查看在训练集上的评价预测精度

## 用训练好的训练集预测平面上每一点的输出##
myML.Neur.plotSamples(train_data,2,instance=clf)


# -------------神经网络模型：用于 iris 模型-----------！！！待升级为多个标签
from matplotlib.colors import ListedColormap
from sklearn import neural_network
## 加载数据集
iris = myML.DataPre.load_datasets("iris")# 使用 scikit-learn  自带的 iris 数据集
X=iris.data[:,0:2] # 使用前两个特征，方便绘图
Y=iris.target # 标记值
data=np.hstack((X,Y.reshape(Y.size,1)))
np.random.seed(0)
np.random.shuffle(data) # 混洗数据。因为默认的iris 数据集：前50个数据是类别0，中间50个数据是类别1，末尾50个数据是类别2.混洗将打乱这个顺序
X=data[:,:-1]
Y=data[:,-1]
train_x=X[:-30]
test_x=X[-30:] # 最后30个样本作为测试集
train_y=Y[:-30]
test_y=Y[-30:]


def plot_classifier_predict_meshgrid(ax,clf,x_min,x_max,y_min,y_max):
      '''
     绘制 MLPClassifier 的分类结果
    :param ax:  Axes 实例，用于绘图
    :param clf: MLPClassifier 实例
    :param x_min: 第一维特征的最小值
    :param x_max: 第一维特征的最大值
    :param y_min: 第二维特征的最小值
    :param y_max: 第二维特征的最大值
    :return: None
      '''
      plot_step = 0.02 # 步长
      xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
      Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
      ax.contourf(xx, yy, Z, cmap=plt.cm.Paired) # 绘图

def plot_samples(ax, x, y):
    '''
      绘制二维数据集
      :param ax:  Axes 实例，用于绘图
      :param x: 第一维特征
      :param y: 第二维特征
      :return: None
    '''
    n_classes = 3
    plot_colors = "bry"  # 颜色数组。每个类别的样本使用一种颜色
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(x[idx, 0], x[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.Paired)  # 绘图


# 使用 MLPClassifier 预测调整后的 iris 数据集
classifier=neural_network.MLPClassifier(activation='logistic',max_iter=10000)

classifier.fit(train_x,train_y)

train_score=classifier.score(train_x,train_y)
test_score=classifier.score(test_x,test_y)

x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
plot_samples(ax,train_x,train_y)
ax.legend(loc='best')
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_title("train score:%f;test score:%f"%(train_score,test_score))
plt.show()



def mlpclassifier_iris_hidden_layer_sizes():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 hidden_layer_sizes 的影响

        :return: None
        '''
        fig=plt.figure()
        hidden_layer_sizes=[(10,),(30,),(100,),(5,5),(10,10),(30,30)] # 候选的 hidden_layer_sizes 参数值组成的数组
        for itx,size in enumerate(hidden_layer_sizes):
            ax=fig.add_subplot(2,3,itx+1)
            classifier=neural_network.MLPClassifier(activation='logistic',max_iter=10000
                ,hidden_layer_sizes=size)
            classifier.fit(train_x,train_y)
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("layer_size:%s;train score:%f;test score:%f"
                %(size,train_score,test_score))
        plt.show()

def mlpclassifier_iris_ativations():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 activation 的影响

        :return:  None
        '''
        fig=plt.figure()
        ativations=["logistic","tanh","relu"] # 候选的激活函数字符串组成的列表
        for itx,act in enumerate(ativations):
            ax=fig.add_subplot(1,3,itx+1)
            classifier=neural_network.MLPClassifier(activation=act,max_iter=10000,
                hidden_layer_sizes=(30,))
            classifier.fit(train_x,train_y)
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("activation:%s;train score:%f;test score:%f"
                %(act,train_score,test_score))
        plt.show()

def mlpclassifier_iris_algorithms():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 algorithm 的影响

        :return: None
        '''
        fig=plt.figure()
        algorithms=["l-bfgs","sgd","adam"] # 候选的算法字符串组成的列表
        for itx,algo in enumerate(algorithms):
            ax=fig.add_subplot(1,3,itx+1)
            classifier=neural_network.MLPClassifier(activation="tanh",max_iter=10000,
                hidden_layer_sizes=(30,),algorithm=algo)
            classifier.fit(train_x,train_y)
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("algorithm:%s;train score:%f;test score:%f"%(algo,train_score,test_score))
        plt.show()

def mlpclassifier_iris_eta():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的学习率的影响

        :return: None
        '''
        fig=plt.figure()
        etas=[0.1,0.01,0.001,0.0001] # 候选的学习率值组成的列表
        for itx,eta in enumerate(etas):
            ax=fig.add_subplot(2,2,itx+1)
            classifier=neural_network.MLPClassifier(activation="tanh",max_iter=1000000,
            hidden_layer_sizes=(30,),algorithm='sgd',learning_rate_init=eta)
            classifier.fit(train_x,train_y)
            iter_num=classifier.n_iter_
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("eta:%f;train score:%f;test score:%f;iter_num:%d"
                %(eta,train_score,test_score,iter_num))
        plt.show()

