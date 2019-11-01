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
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector() # 向量型回测类
#------------------------------------------------------------

# ---获得数据
Path = "C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
CJSecurities = pd.read_csv(Path + '\\CJSecurities.csv', index_col=1, parse_dates=True)
CJSecurities = CJSecurities.iloc[:, 1:]
data0 = CJSecurities

class ABCStrategy(myBT.bt.Strategy):
    # ---设定参数，必须写params，以self.params.Para0索引，可用于优化，内部必须要有逗号
    params = (('Para0', 15),)

    # ---只开头执行一次
    def __init__(self):
        # print("init", self)
        self.barscount = 0
        self.smahandle = myBT.addIndi_SMA(self.datas[0], period=self.params.Para0)
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
        if myBT.orderStatusCheck(order, False) == True:
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
myBT.AddBarsData(data0, fromdate=None, todate=None)

# myBT.addstrategy(ABCStrategy)
myBT.optstrategy(ABCStrategy,Para0=range(5,100))

if __name__ == '__main__':  # 这句必须要有
    myBT.run(plot = False)

