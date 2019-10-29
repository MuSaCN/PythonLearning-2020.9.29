# Author:Zhang Yuan
import MyPackage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPackage.MyClass_Path.MyClass_Path()  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure(AddFigure=False)  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()  #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro(AddFigure=False)  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
myBTE = MyPackage.MyClass_BackTestEvent.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyPackage.MyClass_BackTestVector.MyClass_BackTestVector() # 向量型回测类
myWebQD = MyPackage.MyClass_WebCrawler.MyClass_WebQuotesDownload()  #金融行情下载类
#------------------------------------------------------------

# ---获得数据
Path = "C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
CJSecurities = pd.read_csv(Path + '\\CJSecurities.csv', index_col=1, parse_dates=True)
CJSecurities = CJSecurities.iloc[:, 1:]
data0 = CJSecurities

class ABCStrategy(myBTE.bt.Strategy):
    # ---设定参数，必须写params，以self.params.Para0索引，可用于优化，内部必须要有逗号
    params = (('Para0', 15),)

    # ---只开头执行一次
    def __init__(self):
        print("init", len(self))
        self.smahandle = myBTE.addIndi_SMA(0, period=self.params.Para0)
        self.sma = lambda x: self.smahandle[-x]
        self.barscount = 0
        self.close = self.datas[0].close

    # ---每一个Bar迭代执行一次。next()执行完就进入下一个bar
    def next(self):
        print("next sma:", self.sma(0), self.sma(1), self.sma(2))
        if not self.position:
            if self.close[0] > self.sma(0):
                self.buy()
        else:
            if len(self) >= self.barscount + 5:
                self.sell()

    # ---策略每笔订单通知函数。已经进入下一个bar，且在next()之前执行
    def notify_order(self, order):
        if myBTE.orderStatusCheck(order, True) == True:
            self.barscount = len(self)

    # ---策略每笔交易通知函数。已经进入下一个bar，且在notify_order()之后，next()之前执行。
    def notify_trade(self, trade):
        myBTE.tradeStatus(trade, isclosed=False)

    # ---策略加载完会触发此语句
    def stop(self):
        print("stop(): ", myBTE.ValueCash(), self.sma(0), self.sma(1), self.sma(2))

# ---run
from multiprocessing import freeze_support
# ---opt
if __name__ == '__main__':
    freeze_support()

    # ---基础设置
    myBTE = MyPackage.MyClass_BackTestEvent.MyClass_BackTestEvent()  # 回测类
    myBTE.ValueCash(100000)
    myBTE.setcommission(0.001)
    myBTE.AddBarsData(data0, fromdate=None, todate=None)

    # myBTE.addstrategy(GeneralStrategy)
    # myBTE.run(plot=True)

    myBTE.optstrategy(ABCStrategy,Para0=range(5,100))
    myBTE.cerebro.run(maxcpus=None)
    # myBTE.run(maxcpus=None,plot = False)




