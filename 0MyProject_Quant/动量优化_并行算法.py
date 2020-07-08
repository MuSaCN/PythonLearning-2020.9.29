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
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
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
#------------------------------------------------------------


import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myPjMT5.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True)
price = eurusd.close   # 设定价格为考虑收盘价


# 外部参数
holding_end = 1        # 固定为1，或参数不能大
k_end = 300
lag_trade_end = 5      # 参数不能大
cpu_core = 4

# 必须把总结果写成函数，且只能有一个参数，所以参数以列表或元组形式传递
temp = 0 # 用来显示进度
# ---计算信号，仅分析做多信号
def func_buy(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*cpu_core, k_end*holding_end*lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price, k=k, holding=holding, sig_mode="BuyOnly", stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata["buysignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat["BuyOnly"]
    cumRet = out["cumRet"]
    sharpe = out["sharpe"]
    maxDD = out["maxDD"]
    out["k"] = k
    out["holding"] = holding
    out["lag_trade"] = lag_trade
    # ---
    result = pd.DataFrame()  # 要放到里面
    if cumRet > 0 and sharpe > 0 and maxDD < 0.5:
        result = result.append(out, ignore_index=True)
    return result
# ---计算信号，仅分析做空信号
def func_sell(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*cpu_core, k_end*holding_end*lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price, k=k, holding=holding, sig_mode="SellOnly", stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata["sellsignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat["SellOnly"]
    cumRet = out["cumRet"]
    sharpe = out["sharpe"]
    maxDD = out["maxDD"]
    out["k"] = k
    out["holding"] = holding
    out["lag_trade"] = lag_trade
    # ---
    result = pd.DataFrame()  # 要放到里面
    if cumRet > 0 and sharpe > 0 and maxDD < 0.5:
        result = result.append(out, ignore_index=True)
    return result
# ---计算信号，分析做多做空信号
def func_all(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*cpu_core, k_end*holding_end*lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price, k=k, holding=holding, sig_mode="All", stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata["allsignal"], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat["TotalTrade"]
    cumRet = out["cumRet"]
    sharpe = out["sharpe"]
    maxDD = out["maxDD"]
    out["k"] = k
    out["holding"] = holding
    out["lag_trade"] = lag_trade
    # ---
    result = pd.DataFrame()  # 要放到里面
    if cumRet > 0 and sharpe > 0 and maxDD < 0.5:
        result = result.append(out, ignore_index=True)
    return result



# 多进程必须要在这里写
if __name__ == '__main__':
    # 设定并行参数
    para = [(k, holding, lag_trade) for k in range(1, k_end + 1) for holding in range(1, holding_end + 1) for lag_trade in range(1,lag_trade_end + 1)]
    # ---分析做多信号
    import timeit
    t0 = timeit.default_timer()
    # 必须要写在里面
    out_buy = myBTV.multi_processing(func_buy , para, core_num=cpu_core)
    # 由于out结果为list，需要分开添加
    result_buy = []
    for i in out_buy:
        result_buy.append(i)
    result_buy = pd.concat(result_buy, ignore_index=True)  # 可以自动过滤None
    t1 = timeit.default_timer()
    print("\n",'BUY multi processing 耗时为：', t1 - t0)  # 耗时为：133.3435982
    folder = __mypath__.get_desktop_path() + "\\__动量研究__"
    __mypath__.makedirs(folder, True)
    result_buy.to_excel(folder + "\\result_buy.xlsx")
    # ---分析做空信号
    t0 = timeit.default_timer()
    out_sell = myBTV.multi_processing(func_sell, para, core_num=cpu_core)
    result_sell = []
    for i in out_sell:
        result_sell.append(i)
    result_sell = pd.concat(result_sell, ignore_index=True)  # 可以自动过滤None
    t1 = timeit.default_timer()
    print("\n", 'SELL multi processing 耗时为：', t1 - t0)  # 耗时为：133.3435982
    folder = __mypath__.get_desktop_path() + "\\__动量研究__"
    __mypath__.makedirs(folder, True)
    result_sell.to_excel(folder + "\\result_sell.xlsx")
    # ---分析多空都做信号
    t0 = timeit.default_timer()
    out_all = myBTV.multi_processing(func_all, para, core_num=cpu_core)
    result_all = []
    for i in out_all:
        result_all.append(i)
    result_all = pd.concat(result_all, ignore_index=True)  # 可以自动过滤None
    t1 = timeit.default_timer()
    print("\n", 'ALL multi processing 耗时为：', t1 - t0)  # 耗时为：133.3435982
    folder = __mypath__.get_desktop_path() + "\\__动量研究__"
    __mypath__.makedirs(folder, True)
    result_all.to_excel(folder + "\\result_all.xlsx")














