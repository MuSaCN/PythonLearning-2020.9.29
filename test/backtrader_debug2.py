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
__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\test")  #路径类
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
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myWebQD = MyPackage.MyClass_WebCrawler.MyClass_WebQuotesDownload()  #金融行情下载类
#------------------------------------------------------------

# ---获得数据
Path = "C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
CJSecurities = pd.read_csv(Path + '\\CJSecurities.csv', index_col=1, parse_dates=True)
CJSecurities = CJSecurities.iloc[:, 1:]
data0 = CJSecurities["2015"]

# ---基础设置
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myBT.ValueCash(2000)
myBT.AddBarsData(data0,fromdate=None,todate=None)

# ---策略函数
@myBT.OnInit
def __init__():
    print(myBT.position())
    print("init检测无仓位", not myBT.position())

# ---增加策略
order = []
@myBT.OnNext
def next():
    global order
    if myBT.bars_executed() == 2:
        print(myBT.bars_executed())
        print("未交易无仓位", not myBT.position())
        order.append(myBT.buy())
        print("买入后无仓位未迭代", not myBT.position())
    if myBT.bars_executed() == 3:
        print(myBT.bars_executed())
        print("未卖前无仓位", not myBT.position())
        order.append(myBT.sell())
        print("卖出后无仓位未迭代", not myBT.position())
    if myBT.bars_executed() == 4:
        print(myBT.bars_executed())
        print("迭代交易完无仓位", not myBT.position())

myBT.addstrategy()
# ---运行
myBT.run(plot = False)





import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


# Create a Stratey
class TestStrategy(bt.Strategy):

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

    # ---订单创建会触发执行此语句，订单未执行完会多次执行此语句
    def notify_order(self, order):
        print("notify_order执行")
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        print("notify_order执行1")
        if order.status in [order.Completed]:
            if order.isbuy():
                print('BUY 执行, %.2f' % order.executed.price)
            elif order.issell():
                print('SELL 执行, %.2f' % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')
        # Write down: no pending order
        self.order = None

    def next(self):
        print(len(self))
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        if not self.position:
            if self.dataclose[0] < self.dataclose[-1]:
                    if self.dataclose[-1] < self.dataclose[-2]:
                        print('BUY 创建, %.2f' % self.dataclose[0])
                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()
        else:
            if len(self) >= (self.bar_executed + 5):
                print('SELL 创建, %.2f' % self.dataclose[0])
                self.order = self.sell()


# Create a cerebro entity
cerebro = bt.Cerebro()

# Add a strategy
cerebro.addstrategy(TestStrategy)
if "openinterest" not in data0.columns:
    data0['openinterest'] = 0
data = bt.feeds.PandasData(dataname=data0)
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)
cerebro.run()

