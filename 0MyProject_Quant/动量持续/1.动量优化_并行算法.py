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

'''
说明：
# 参数优化部分，需要专门设定训练集和测试集。由于参数较多，不可能都通过图示。所以，通过训练集来计算出各个参数下策略结果，安全起见保存结果到硬盘。
# 再根据训练集参数优化的结果，计算对应参数下测试集策略结果，把结果保存到硬盘。
# 整合两个结果到一张表格。
# 需要注意的是，由于 训练集和测试集 信号计算时依赖的数据集不同，所以要设定两个函数。
'''

################# 设置数据，区分训练集和测试集 ########################
import warnings
warnings.filterwarnings('ignore')

# ---获取数据
eurusd = myPjMT5.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
eurusd_train = eurusd.loc[:"2014-12-31"]
eurusd_test = eurusd.loc["2015-01-01":]
price_train = eurusd_train.Close
price_test = eurusd_test.Close

################# 设置参数，设置范围的最大值 ##########################
# 外部参数
paranames = ["k", "holding", "lag_trade"]  # 顺序不能搞错了，要与信号函数中一致
k_end = 350             # 动量向左参数
holding_end = 1         # 可以不同固定为1
lag_trade_end = 1       # 参数不能大

################# 信号函数部分，或多个函数、或多个参数 #####################
# 必须把总结果写成函数，且只能有一个参数，所以参数以列表或元组形式传递。内部参数有的要依赖于外部。
temp = 0 # 用来显示进度
# ---训练集 计算信号
def signalfunc_train(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3]  # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    sig_mode, signalname, tradename = myBTV.get_direct_str_index(trade_direct)
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp * cpu_core, k_end * holding_end * lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_train, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata[signalname], price_DataFrame=eurusd_train, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, paranames)
    return result
# ---测试集 计算信号
def signalfunc_test(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3]  # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    sig_mode, signalname, tradename = myBTV.get_direct_str_index(trade_direct)
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp * cpu_core, k_end * holding_end * lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_test, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata[signalname], price_DataFrame=eurusd_test, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, paranames)
    return result

# ---训练集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_train(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3] # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    sig_mode, signalname, tradename = myBTV.get_direct_str_index(trade_direct)
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*cpu_core, k_end*holding_end*lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_train, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[signalname], price_DataFrame=eurusd_train, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, paranames)
    return result
# ---测试集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_test(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3] # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    sig_mode, signalname, tradename = myBTV.get_direct_str_index(trade_direct)
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*cpu_core, k_end*holding_end*lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_test, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[signalname], price_DataFrame=eurusd_test, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, paranames)
    return result

################# 多进程执行函数 ########################################
cpu_core = 4
# ---多进程必须要在这里执行
if __name__ == '__main__':
    # ---设定并行参数
    para_buyonly = [(k, holding, lag_trade, "BuyOnly") for k in range(1, k_end + 1) for holding in
                    range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]
    para_sellonly = [(k, holding, lag_trade, "SellOnly") for k in range(1, k_end + 1) for holding
                     in range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]
    para_all = [(k, holding, lag_trade, "All") for k in range(1, k_end + 1) for holding in
                range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]

    # ---分析训练集(并行)
    folder = __mypath__.get_desktop_path() + "\\__动量研究(test)__"
    buyfilepath = folder + "\\动量_Buy.xlsx"
    sellfilepath = folder + "\\动量_Sell.xlsx"
    allfilepath = folder + "\\动量_All.xlsx"
    myBTV.run_train(signalfunc_NoRepeatHold_train, para_buyonly, buyfilepath, cpu_core)
    myBTV.run_train(signalfunc_NoRepeatHold_train, para_sellonly, sellfilepath, cpu_core)
    myBTV.run_train(signalfunc_NoRepeatHold_train, para_all, allfilepath, cpu_core)

    # ---分析测试集(并行)
    myBTV.run_test(signalfunc_NoRepeatHold_test, buyfilepath, paranames, cpu_core)
    myBTV.run_test(signalfunc_NoRepeatHold_test, sellfilepath, paranames, cpu_core)
    myBTV.run_test(signalfunc_NoRepeatHold_test, allfilepath, paranames, cpu_core)














