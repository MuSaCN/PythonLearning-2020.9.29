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
__mypath__ = MyPath.MyClass_Path("\\0MyProject_Quant")  # 路径类
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
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
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
myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
myPjMT5 = MyProject.MT5_MLLearning() # MT5机器学习项目类
#------------------------------------------------------------

#%% ###################################
# ---获取数据
eurusd = myPjMT5.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True)
close = eurusd["close"]
rate1 = eurusd["rate"]

#%%
# ---数据解读
eurusd.dtypes
myDA.describe(eurusd)
data = pd.DataFrame({"Open":eurusd["open"], "High":eurusd["high"],
                     "Low":eurusd["low"], "Close": eurusd["close"]}, index = eurusd["time"])
myDA.indi.candle_ohlc(data)

# ---波动率分析
myDA.tsa_auto_test(rate1[1:])
myDA.tsa_auto_ARIMA(rate1[1:])
myDA.tsa_auto_ARCH(rate1[1:])

# ---序列自相关系数分析：1期波动与其滞后的相关系数曲线
myDA.tsa.plot_selfcorrelation(rate1,count=500)

# ---序列惯性分析：1期波动与n期波动的相关系数
myDA.tsa.plot_inertia(eurusd["close"],n_start=1,n_end=500,shift=1)

#%%
## ---转换技术指标
rate1 = eurusd["rate"]
close = eurusd["close"]
import talib
rsi = talib.RSI(close,timeperiod=13)
myDA.tsa_auto_test(rsi.dropna())  # 平稳过程，可以分析


rate1.corr(rsi.shift(-1), method="pearson")  #！！！当天波动与明天rsi指标关系 0.329
rate1.corr(rsi.shift(-1), method="kendall")  #！！！当天波动与明天rsi指标关系 0.329
rate1.corr(rsi.shift(-1), method="spearman") #！！！当天波动与明天rsi指标关系 0.329

rsi_rate = rsi.pct_change(periods=1)
rate1.corr(rsi_rate.shift(1), method="pearson")


# 两天波动的信息包括今天和昨天，所以今天的波动与两天的波动、昨天的波动与两天的波动 相关性都大。
# (其实无意义，两天波动可以通过运用公式，把一天的波动作为变量来算出。)
rate2 = close.pct_change(periods=2)
rate1.corr(rate2.shift(-1), method="pearson")
rate2.corr(rate1.shift(1), method="pearson")

#%% ##############################################
# 以时间向量的方式，获取指定时间之前(包括指定时间)的n个波动率
data0 = myPjMT5.getvolatility_beforetime(rate1.index,"EURUSD","TIMEFRAME_H1",count=5)

#%% ##############################################
# 获取非共线性的技术指标
rate1 = eurusd["rate"]
close = eurusd["close"]
import talib
rsi = talib.RSI(close,timeperiod=13)
myDA.tsa_auto_test(rsi.dropna())  # 平稳过程，可以分析







