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
myWebQD = MyWebCrawler.MyClass_QuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
#------------------------------------------------------------

myMT5 = MyMql.MyClass_ConnectMT5(connect=True)
rates = myMT5.copy_rates_from("EURUSD",myMT5.mt5.TIMEFRAME_H1,[2019,11,1],10)
rates = myMT5.copy_rates_from_pos("GBPUSD",myMT5.mt5.TIMEFRAME_D1,1,10)
rates = myMT5.copy_rates_range("EURUSD",myMT5.mt5.TIMEFRAME_H1,[2019,10,1],[2019,10,3])
rates1 = myMT5.rates_to_DataFrame(rates,True)
myMT5.datatime_zone_offset(rates1)

ticks = myMT5.copy_ticks_from("EURUSD",myMT5.mt5.COPY_TICKS_ALL,[2019,10,20],1000)
ticks = myMT5.copy_ticks_range("EURUSD",myMT5.mt5.COPY_TICKS_ALL,[2019,10,10],[2019,10,11])
ticks1 = myMT5.ticks_to_DataFrame(ticks,True)
myMT5.datatime_zone_offset(ticks1)






