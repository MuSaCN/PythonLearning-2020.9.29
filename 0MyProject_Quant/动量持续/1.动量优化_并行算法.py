# Author:Zhang Yuan
import warnings
warnings.filterwarnings('ignore')
#
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
# 由于并行运算的原理，参数分为 策略参数 + 非策略参数
'''

#%% ################# 设置参数，设置范围的最大值 ##########################
# 策略参数(需写在这里)，按顺序保存在 para 的前面
strategy_para_names = ["k", "holding", "lag_trade"]  # 顺序不能搞错了，要与信号函数中一致
k_end = 400             # 动量向左参数
holding_end = 10        # 持有期参数，可以不同固定为1
lag_trade_end = 1       # 信号出现滞后交易参数，参数不能大

#%% ################# 信号函数部分，或多个函数、或多个参数 #####################
temp = 0  # 用来显示进度，必须放在这里
# 必须把总结果写成函数，且只能有一个参数，所以参数以列表或元组形式传递。内部参数有的要依赖于外部。

# ---训练集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_train(para):
    # 策略参数
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{} train: ".format(temp * cpu_core, k_end * holding_end * lag_trade_end), para, end="", flush=True)
    # 非策略参数
    trade_direct = para[-3] # "BuyOnly","SellOnly","All"
    symbol = para[-2]
    timeframe = para[-1]
    # 获取数据
    date_from, date_to = myPjMT5.get_date_range(timeframe) # 不同时间框架加载的时间范围不同
    data_total = myPjMT5.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    data_train, data_test = myPjMT5.get_train_test(data_total, train_scale=0.8)
    # 不同交易方向下，数据字符串索引
    sig_mode, signalname, tradename = myBTV.get_direct_str_index(trade_direct)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(data_train.Close, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[signalname], price_DataFrame=data_train, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, strategy_para_names)
    return result

# ---测试集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_test(para):
    # 策略参数
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{} test: ".format(temp * cpu_core, k_end * holding_end * lag_trade_end), para, end="", flush=True)
    # 非策略参数
    trade_direct = para[-3]  # "BuyOnly","SellOnly","All"
    symbol = para[-2]
    timeframe = para[-1]
    # 获取数据
    date_from, date_to = myPjMT5.get_date_range(timeframe) # 不同时间框架加载的时间范围不同
    data_total = myPjMT5.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    data_train, data_test = myPjMT5.get_train_test(data_total, train_scale=0.8)
    # 不同交易方向下，数据字符串索引
    sig_mode, signalname, tradename = myBTV.get_direct_str_index(trade_direct)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(data_train.Close, k=k, holding=holding, sig_mode=sig_mode, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[signalname], price_DataFrame=data_test, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, strategy_para_names)
    return result

################# 多进程执行函数 ########################################
cpu_core = 11
# ---多进程必须要在这里执行
if __name__ == '__main__':
    # ---非策略参数：
    # 方向参数："BuyOnly" "SellOnly" "All"，保存在 para 的 -3 位置
    direct_para = ["BuyOnly", "SellOnly"] # direct_para = ["BuyOnly", "SellOnly", "All"]
    # symbol、timeframe 参数设置在 -2、-1 的位置
    symbol_list = myPjMT5.get_all_symbol_name().tolist()
    timeframe_list = ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6",
                      "TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1",
                      "TIMEFRAME_M30","TIMEFRAME_M20","TIMEFRAME_M15","TIMEFRAME_M12",
                      "TIMEFRAME_M10","TIMEFRAME_M6","TIMEFRAME_M5","TIMEFRAME_M4",
                      "TIMEFRAME_M3","TIMEFRAME_M2","TIMEFRAME_M1"]
    # ---开始并行运算
    for timeframe in timeframe_list:
        if timeframe in ["TIMEFRAME_D1", "TIMEFRAME_H12","TIMEFRAME_H8"]:
            continue
        finish_symbol = []
        for symbol in symbol_list:
            if symbol == "EURUSD" and timeframe in ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6","TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1"]:
                continue
            if timeframe == "TIMEFRAME_H6" and symbol in ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 'AUDUSD', 'AUDNZD', 'AUDCAD', 'AUDCHF', 'AUDJPY','CHFJPY', 'EURGBP', 'EURAUD', 'EURCHF', 'EURJPY', 'EURNZD', 'EURCAD', 'GBPCHF', 'GBPJPY', 'CADCHF', 'CADJPY', 'EURTRY', 'GBPNZD', 'USDDKK', 'USDHKD', 'USDNOK', 'USDSEK', 'USDSGD', 'USDTRY', 'GBPAUD', 'GBPCAD', 'NZDJPY', 'NZDUSD']:
                finish_symbol.append(symbol)
                continue
            # 设置输出目录：one symbol + one timeframe + three direct --> one folder
            folder = __mypath__.get_desktop_path() + "\\_动量研究\\{}.{}".format(symbol, timeframe)
            # 仅做多、仅做空、多空都做，保存在一个目录下
            for direct in direct_para:
                # 设定并行参数，只需要指定策略参数的范围即可
                para_muilt = [(k, holding, lag_trade, direct, symbol, timeframe) for k in range(1, k_end + 1) for holding in range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]
                filepath = folder + "\\动量_{}.xlsx".format(direct)
                # 分析训练集(并行)，会把参数优化结果生成文档。
                myBTV.run_train(signalfunc_NoRepeatHold_train, para_muilt, filepath, cpu_core)
                # 分析测试集(并行)，会内部解析训练集文档中的参数。
                # myBTV.run_test(signalfunc_NoRepeatHold_test, filepath, strategy_para_names, [direct,symbol,timeframe],cpu_core)
            finish_symbol.append(symbol)
            print("finished:", timeframe, finish_symbol)











