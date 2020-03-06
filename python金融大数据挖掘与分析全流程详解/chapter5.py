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
__mypath__ = MyPath.MyClass_Path("\\python金融大数据挖掘与分析全流程详解")  # 路径类
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
myWebC = MyWebCrawler.MyClass_WebCrawler()  # 综合网络爬虫类
mySQL = MyDatabase.MyClass_MySQL(connect=False)  # MySQL类
#------------------------------------------------------------

# 5.1 数据去重及清洗优化
company = "阿里巴巴"
text = myWebC.get("http://finance.sina.com.cn/stock/hkstock/marketalerts/2020-03-06/doc-iimxyqvz8239967.shtml").text
print(text)
myWebC.findall(company[0] + '.{0,5}' + company[-1], text)


myWebC.news_baidu("阿里巴巴",rtt=1,checkhref=False,word_href=None)
myWebC.news_baidu("阿里巴巴",rtt=1,checkhref=True,word_href=None)
myWebC.news_baidu("阿里巴巴",rtt=1,checkhref=True,word_href=None,database="quant.news")


myWebC.news_sogou("阿里巴巴")
myWebC.news_sina("阿里巴巴")


