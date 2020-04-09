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
__mypath__ = MyPath.MyClass_Path("\\DeepLearningWithPython")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
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
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDatabase.MyClass_MySQL(connect=False)  # MySQL类
mySQLAPP = MyDatabase.MyClass_SQL_APPIntegration()  # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
myFactorD = MyQuant.MyClass_Factor_Detection()  # 因子检测类
myKeras = MyDeepLearning.MyClass_Keras()  # Keras综合类
#------------------------------------------------------------


#%%
# 函数式 API 实现
# 简单的例子
from keras import Input
input_tensor = Input(shape=(64,))

from keras import layers
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

# Model 类将输入张量和输出张量转换为一个模型
from keras.models import  Model
model = Model(input_tensor, output_tensor)

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# 生成用于训练的虚构 Numpy 数据
import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)


#%%
# 用函数式 API 实现双输入问答模型
from keras.models import Model
from keras import layers
from keras import Input
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
# 文本输入是一个长度可变的整数序列。注意，你可以选择对输入进行命名
text_input = Input(shape=(None,), dtype='int32', name='text')
# 将输入嵌入长度为 64 的向量
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
# 利用 LSTM 将向量编码为单个向量
encoded_text = layers.LSTM(32)(embedded_text)
# 对问题进行相同的处理（使用不同的层实例）
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
# 将编码后的问题和文本连接起来
concatenated = layers.concatenate([encoded_text, encoded_question],axis=-1)
# 在上面添加一个softmax 分类器
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
# 在模型实例化时，指定两个输入和输出
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# 将数据输入到多输入模型中
import numpy as np
num_samples = 1000
max_length = 100
# 生成虚构的 Numpy 数据
text = np.random.randint(1, text_vocabulary_size,size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size,size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
import keras
# 回答是 one-hot 编码的，不是整数
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)
# 使用输入组成的列表来拟合
model.fit([text, question], answers, epochs=10, batch_size=128)
# 使用输入组成的字典来拟合（只有对输入进行命名之后才能用这种方法）
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)



#%%
# 用函数式 API 实现一个三输出模型




