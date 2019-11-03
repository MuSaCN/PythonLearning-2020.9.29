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
__mypath__ = MyPath.MyClass_Path()  # 路径类
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
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
#------------------------------------------------------------

myMT5 = MyMql.MyClass_ConnectMT5(connect=True)
rates = myMT5.MT5CopyRatesFrom("EURUSD",myMT5.mt5.MT5_TIMEFRAME_H1,[2019,11,1],10)
rates = myMT5.MT5CopyRatesFromPos("GBPUSD",myMT5.mt5.MT5_TIMEFRAME_D1,1,10)
rates = myMT5.MT5CopyRatesRange("EURUSD",myMT5.mt5.MT5_TIMEFRAME_H1,[2019,10,1],[2019,10,3])
rates1 = myMT5.RatesToDataFrame(rates,False)
myMT5.DataTimeZoneOffset(rates1)

ticks = myMT5.MT5CopyTicksFrom("EURUSD",myMT5.mt5.MT5_COPY_TICKS_ALL,[2019,10,20],1000)
ticks = myMT5.MT5CopyTicksRange("EURUSD",myMT5.mt5.MT5_COPY_TICKS_ALL,[2019,10,10],[2019,10,11])
ticks1 = myMT5.TicksToDataFrame(ticks,True)
myMT5.DataTimeZoneOffset(ticks1)


eurusd_ticks = myMT5.MT5CopyTicksFrom("EURUSD",myMT5.mt5.MT5_COPY_TICKS_ALL,[2019, 4, 1, 0], 1000)
audusd_ticks = myMT5.MT5CopyTicksRange("AUDUSD", myMT5.mt5.MT5_COPY_TICKS_ALL,(2019, 4, 1, 13), (2019, 4, 2, 13))

# get bars from different symbols in a number of ways
eurusd_rates = myMT5.MT5CopyRatesFrom("EURUSD", myMT5.mt5.MT5_TIMEFRAME_M1, (2019, 4, 5, 15), 1000)
audusd_rates = myMT5.MT5CopyRatesFromPos("AUDUSD", myMT5.mt5.MT5_TIMEFRAME_M1, 0, 1000)
gbpusd_rates = myMT5.MT5CopyRatesRange("GBPUSD", myMT5.mt5.MT5_TIMEFRAME_M1, (2019, 4, 1, 13), (2019, 4, 2, 13))





