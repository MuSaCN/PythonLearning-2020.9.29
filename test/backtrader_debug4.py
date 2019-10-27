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
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  # 事件驱动型回测
myBTV = MyPackage.MyClass_BackTestVector.MyClass_BackTestVector()  # 向量化回测
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
myBT.setcommission(0.001)
myBT.AddBarsData(data0,fromdate=None,todate=None)

# ---策略开始
@myBT.OnInit
def __init__():
    print("init检测无仓位 = ", not myBT.position())

# ---策略递归
order = []; barscount = [0]
@myBT.OnNext
def next():
    print(myBT.bars_executed)
    if not myBT.position():
        if (myBT.close(0) < myBT.close(1)) and (myBT.close(1)< myBT.close(2)) and myBT.bars_executed>=3:
            order.append(myBT.buy())
    else:
        if myBT.bars_executed >= barscount[0]+5:
            order.append(myBT.sell())

# ---策略订单触发订单通知，会在下一个bar的next()之前执行
@myBT.OnNotify_Order
def notify_order():
    if myBT.orderStatusCheck(myBT.order_noti,True) == False:
        return
    else:
        # 必须记录在这里，因为执行在这里
        barscount[0] = myBT.bars_executed


myBT.addstrategy()
# ---运行
myBT.run(plot = True)





import backtrader as bt

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close

    def notify_order(self, order):
        print(len(self),"  notify_order start")
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            self.bar_executed = len(self)
        print(len(self),"  notify_order end")

    # ---在每次notify_order()中order成功后，会触发这个函数，用于处理每笔trade交易行为的信息。比如是否交易关闭、交易利润、交易净利润...
    def notify_trade(self, trade):
        print(len(self),"  notify_trade start")
        if not trade.isclosed:
            return
        print('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))
        print(len(self), "  notify_trade end")

    def next(self):
        if not self.position:
            if self.dataclose[0] < self.dataclose[-1]:
                    if self.dataclose[-1] < self.dataclose[-2]:
                        self.buy()
                        print('BUY CREATE, %.2f' % self.dataclose[0])
        else:
            if len(self) >= (self.bar_executed + 5):
                self.sell()
                print('SELL CREATE, %.2f' % self.dataclose[0])

cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)
if "openinterest" not in data0.columns:  # 检测是否需要增加'openinterest'列
    data0['openinterest'] = 0
data = bt.feeds.PandasData(dataname=data0)
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.run()



