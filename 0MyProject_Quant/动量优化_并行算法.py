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


import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myPjMT5.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
price = eurusd.Close   # 设定价格为考虑收盘价
price_train = price
price_test = price
# price_train = price.loc[:"2018-12-31"]
# price_test = price.loc["2019-01-01":]

# 外部参数
holding_end = 5         # 可以不同固定为1
k_end = 100             # 动量向左参数
lag_trade_end = 1       # 参数不能大



# 必须把总结果写成函数，且只能有一个参数，所以参数以列表或元组形式传递。内部参数有的要依赖于外部。
temp = 0 # 用来显示进度
# ---训练集 计算信号
def signalfunc_train(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3]  # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    if trade_direct == "BuyOnly":
        sig_mode, signalname, tradename = "BuyOnly", "buysignal", "BuyOnly"
    elif trade_direct == "SellOnly":
        sig_mode, signalname, tradename = "SellOnly", "sellsignal", "SellOnly"
    elif trade_direct == "All":
        sig_mode, signalname, tradename = "All", "allsignal", "AllTrade"
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp * cpu_core, k_end * holding_end * lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_train, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata[signalname], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat[tradename]
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
# ---训练集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_train(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3] # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    if trade_direct == "BuyOnly":
        sig_mode, signalname, tradename = "BuyOnly", "buysignal", "BuyOnly"
    elif trade_direct == "SellOnly":
        sig_mode, signalname, tradename = "SellOnly", "sellsignal", "SellOnly"
    elif trade_direct == "All":
        sig_mode, signalname, tradename = "All", "allsignal", "AllTrade"
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*cpu_core, k_end*holding_end*lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_train, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[signalname], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat[tradename]
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
# ---测试集 计算信号
def signalfunc_test(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3]  # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    if trade_direct == "BuyOnly":
        sig_mode, signalname, tradename = "BuyOnly", "buysignal", "BuyOnly"
    elif trade_direct == "SellOnly":
        sig_mode, signalname, tradename = "SellOnly", "sellsignal", "SellOnly"
    elif trade_direct == "All":
        sig_mode, signalname, tradename = "All", "allsignal", "AllTrade"
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp * cpu_core, k_end * holding_end * lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_test, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata[signalname], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat[tradename]
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
# ---测试集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_test(para):
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3] # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    if trade_direct == "BuyOnly":
        sig_mode, signalname, tradename = "BuyOnly", "buysignal", "BuyOnly"
    elif trade_direct == "SellOnly":
        sig_mode, signalname, tradename = "SellOnly", "sellsignal", "SellOnly"
    elif trade_direct == "All":
        sig_mode, signalname, tradename = "All", "allsignal", "AllTrade"
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*cpu_core, k_end*holding_end*lag_trade_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price_test, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[signalname], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat[tradename]
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


cpu_core = 4
# ---多进程执行函数，优化结果生成文档
def run_train(func, para, name):
    import timeit
    t0 = timeit.default_timer()
    out = myBTV.multi_processing(func, para, core_num=cpu_core)
    # 由于out结果为list，需要分开添加
    result = []
    for i in out:
        result.append(i)
    result = pd.concat(result, ignore_index=True)  # 可以自动过滤None
    t1 = timeit.default_timer()
    print("\n", '{} multi processing 耗时为：'.format(name), t1 - t0)  # 耗时为：670.1083746
    folder = __mypath__.get_desktop_path() + "\\__动量研究__"
    __mypath__.makedirs(folder, True)
    result.to_excel(folder + "\\{}.xlsx".format(name))
# ---测试集执行，会内部解析参数
def run_test(func, name):
    # ---解析参数
    folder = __mypath__.get_desktop_path() + "\\__动量研究__"
    trainpath = folder + "\\{}.xlsx".format(name)
    paranames = ["k", "holding", "lag_trade"]  # 顺序不能搞错了
    paralist = myBTV.parse_opt_xlsx(trainpath, paranames)
    # ---执行，生成测试集
    name_test = name+"_测试集"
    testpath = folder + "\\{}.xlsx".format(name_test)
    run_train(func, paralist, name_test)
    # ---合并两个数据
    myBTV.concat_opt_xlsx(trainpath, testpath, paranames)

# ---多进程必须要在这里执行
if __name__ == '__main__':
    # ---设定并行参数
    para_buyonly = [(k, holding, lag_trade, "BuyOnly") for k in range(1, k_end + 1) for holding in
                    range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]
    para_sellonly = [(k, holding, lag_trade, "SellOnly") for k in range(1, k_end + 1) for holding in
                     range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]
    para_all = [(k, holding, lag_trade, "All") for k in range(1, k_end + 1) for holding in
                range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]
    # ---分析训练集(并行)
    run_train(signalfunc_train,para_buyonly,"动量_Buy")  # 分析做多信号
    run_train(signalfunc_train,para_sellonly,"动量_Sell") # 分析做空信号
    run_train(signalfunc_train,para_all,"动量_All") # 分析多空信号
    # ---分析测试集(并行)
    run_test(signalfunc_test,"动量_Buy")
    run_test(signalfunc_test, "动量_Sell")
    run_test(signalfunc_test, "动量_All")













