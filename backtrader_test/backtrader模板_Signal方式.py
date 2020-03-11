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
myWebQD = MyWebCrawler.MyClass_QuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector() # 向量型回测类
#------------------------------------------------------------

# ---获得数据
Path = "C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
CJSecurities = pd.read_csv(Path + '\\CJSecurities.csv', index_col=1, parse_dates=True)
CJSecurities = CJSecurities.iloc[:, 1:]
data0 = CJSecurities

# ---Signal信号方式构建快速策略
class MySignal(myBT.bt.Indicator):
    lines = ('signal',) # 设定返回的lines
    params = (('period', 30),)
    def __init__(self):
        self.lines.signal = self.data - myBT.add_indi_sma(self.data, period=self.params.period)

# ---基础设置
myBT = MyBackTest.MyClass_BackTestEvent()  # 回测类
myBT.setcash(100000)
myBT.setcommission(0.001)
myBT.addsizer(10)
myBT.adddata(data0, fromdate=None, todate=None)
myBT.addanalyzer_all()

if __name__ == '__main__':  # 这句必须要有
    # ---加入到信号中，解析信号进行交易
    # LONGSHORT: 买入卖出信号都接受执行
    # LONG:买入信号执行，卖出信号仅仅将多头头寸平仓，而不反向卖出。
    # SHORT:卖出信号被执行，而买入信号仅仅将空头头寸平仓，而不方向买入。
    myBT.signal_run("LONGSHORT",MySignal,plot=False)
    # ---优化这个方式不可用
    # myBT.OptRun(MySignal,Para0=range(5,100))

myBT.get_analysis('VWR')



