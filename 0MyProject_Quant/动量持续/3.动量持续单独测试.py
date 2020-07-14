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
__mypath__ = MyPath.MyClass_Path("")  # 路径类
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
myDefault = MyDefault.MyClass_Default_Matplotlib()  # 画图恢复默认设置类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDataBase.MyClass_MySQL(connect=False)  # MySQL类
mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration()  # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
myFactorD = MyQuant.MyClass_Factor_Detection()  # 因子检测类
myKeras = MyDeepLearning.MyClass_tfKeras()  # tfKeras综合类
myTensor = MyDeepLearning.MyClass_TensorFlow()  # Tensorflow综合类
myMT5 = MyMql.MyClass_ConnectMT5(connect=False)  # Python链接MetaTrader5客户端类
myPjMT5 = MyProject.MT5_MLLearning()  # MT5机器学习项目类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
说明：
# 单独测试时，在所有的数据集上，不需要专门的设定训练集、测试集，只需要指定画图时训练集的区间即可。
# 显然，测试集可以在训练集之前，也可以在训练集之后。通过图示能更好的观察收益情况。
'''

#%%
########## 单次测试部分 #################
import warnings
warnings.filterwarnings('ignore')

# ---获取数据
eurusd = myPjMT5.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
price = eurusd.Close
# 单独测试不需要把数据集区分训练集、测试集，仅画区间就可以了
train_x0 = pd.Timestamp('2000-01-01 00:00:00')
train_x1 = pd.Timestamp('2014-12-31 00:00:00')


#%%
# ---仅做多分析
holding = 1
k = 100
lag_trade = 1

# ---获取信号数据
signaldata_buy = myBTV.stra.momentum(price, k=k, holding=holding, sig_mode="BuyOnly", stra_mode="Continue")
# 信号分析
outStrat, outSignal = myBTV.signal_quality(signal=signaldata_buy["buysignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)

# 信号分析，不重复持仓
outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata_buy["buysignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)

myBTV.signal_quality_explain()


#%%
# ---仅做空分析
holding = 1
k = 100
lag_trade = 1

# ---获取信号数据
signaldata_sell = myBTV.stra.momentum(price, k=k, holding=holding, sig_mode="SellOnly", stra_mode="Continue")
# 信号分析
outStrat, outSignal = myBTV.signal_quality(signaldata_sell["sellsignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)
# 信号分析，不重复持仓
outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata_sell["sellsignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)
myBTV.signal_quality_explain()


#%%
# ---多空都做分析，相同参数
holding = 1
k = 100
lag_trade = 1

# ---获取信号数据
signaldata_all = myBTV.stra.momentum(price, k=k, holding=holding, sig_mode="All", stra_mode="Continue")
# 信号分析
outStrat, outSignal = myBTV.signal_quality(signaldata_all["allsignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)
# 信号分析，不重复持仓
outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata_all["allsignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)
myBTV.signal_quality_explain()


#%%
# ---多空不同参数合并分析
signal_add = signaldata_buy["buysignal"] + signaldata_sell["sellsignal"]
holding = 1
lag_trade = 5
# 信号分析
outStrat, outSignal = myBTV.signal_quality(signal_add, price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)
# 信号分析，不重复持仓
outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signal_add, price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=True, plotStrat=True, train_x0=train_x0, train_x1=train_x1)
myBTV.signal_quality_explain()



