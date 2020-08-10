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

# ---获取数据
eurusd = myPjMT5.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)

data0 = eurusd

class ABCStrategy(myBT.bt.Strategy):
    # ---设定参数，必须写params，以self.params.Para0索引，可用于优化，内部必须要有逗号
    params = (('Para0', 15),)

    # ---只开头执行一次
    def __init__(self):
        # print("init", self)
        self.barscount = 0
        self.smahandle = myBT.indi.add_indi_sma(self.datas[0], period=self.params.Para0)


        myBT.indi.add_indi_sma(data0, period=15)
        from MyPackage.MyBackTest.BackTestEvent_Indicators import MyClass_BackTest_Indicators
        a = MyClass_BackTest_Indicators()
        a.add_indi_sma(data0, period=15)


        self.sma = lambda x: self.smahandle[-x]
        # open索引
        self.openTemp = self.datas[0].open
        self.open = lambda x: self.openTemp[-x]
        # high索引
        self.highTemp = self.datas[0].high
        self.high = lambda x: self.highTemp[-x]
        # low索引
        self.lowTemp = self.datas[0].low
        self.low = lambda x: self.lowTemp[-x]
        # close索引
        self.closeTemp = self.datas[0].close
        self.close = lambda x: self.closeTemp[-x]
        # datetime.date索引
        # self.timeTemp = self.datas[0].datetime.date
        # self.datetime = lambda x: self.timeTemp[-x]

    # ---每一个Bar迭代执行一次。next()执行完就进入下一个bar
    def next(self):
        if not self.position:
            if self.close(0) > self.sma(0):
                self.buy()
        else:
            if len(self) >= self.barscount + 5:
                self.sell()

    # ---策略每笔订单通知函数。已经进入下一个bar，且在next()之前执行
    def notify_order(self, order):
        if myBT.order_status_check(order, False) == True:
            self.barscount = len(self)

    # ---策略每笔交易通知函数。已经进入下一个bar，且在notify_order()之后，next()之前执行。
    def notify_trade(self, trade):
        pass
        # myBT.tradeStatus(trade, isclosed=False)

    # ---策略加载完会触发此语句
    def stop(self):
        print("stop(): ", self.params.Para0 , self.broker.getvalue(), self.broker.get_cash())


# ---基础设置
myBT = MyBackTest.MyClass_BackTestEvent()  # 回测类
myBT.setcash(100000)

myBT.setcommission(0.001)
myBT.adddata(data0, fromdate=None, todate=None)

myBT.addstrategy(ABCStrategy)
# myBT.optstrategy(ABCStrategy,Para0=range(5,100))

if __name__ == '__main__':  # 这句必须要有
    myBT.run(maxcpus=1 ,plot = True)





