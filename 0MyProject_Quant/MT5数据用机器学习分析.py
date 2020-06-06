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
# myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
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
#------------------------------------------------------------

# ---获取数据
myMT5 = MyMql.MyClass_ConnectMT5(connect=True) # Python链接MetaTrader5客户端类
eurusd = myMT5.copy_rates_range("EURUSD",myMT5.mt5.TIMEFRAME_D1,[2010,1,1,0,0,0],[2020,1,1,0,0,0])
eurusd = myMT5.rates_to_DataFrame(eurusd,True)
eurusd.index = eurusd["time"]
myMT5.shutdown()

# ---数据清洗
eurusd.isnull().sum() # 是否有缺失值
eurusd["rate"] = eurusd["close"].pct_change(periods=1) # 增加一期收盘价收益率

# ---数据解读
eurusd.dtypes
myDA.describe(eurusd)
data = pd.DataFrame({"Open":eurusd["open"], "High":eurusd["high"],
                     "Low":eurusd["low"], "Close": eurusd["close"]}, index = eurusd["time"])
myDA.candle_ohlc(data)

# ---波动率分析
rate = pd.concat([eurusd["rate"], eurusd["rate"].shift(1), eurusd["rate"].shift(2),
                  eurusd["rate"].shift(3),eurusd["rate"].shift(4)], axis=1)
rate.columns = ["rate","rate_s1","rate_s2","rate_s3","rate_s4"]

myDA.tsa_auto_test(rate["rate"][1:])
myDA.tsa_auto_ARIMA(rate["rate"][1:])
myDA.tsa_auto_ARCH(rate["rate"][1:])




