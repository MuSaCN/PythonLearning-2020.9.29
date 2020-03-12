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
__mypath__ = MyPath.MyClass_Path("\\python金融大数据挖掘与分析全流程详解")  # 路径类
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
# myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDatabase.MyClass_MySQL(connect=False)  # MySQL类
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_APPIntegration() # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
#------------------------------------------------------------

# 12.1.1-1 requests库下载文件
url = 'http://images.china-pub.com/ebook8055001-8060000/8057968/shupi.jpg'
myWebR.download(url,"pic.jpg")


# 12.1.1-2 通过pandas获取表格
url = 'http://vip.stock.finance.sina.com.cn/q/go.php/vInvestConsult/kind/dzjy/index.phtml'  # 新浪财经数据中心提供股票大宗交易的在线表格
myWebAPP.read_html(url,to_excel="测试.xlsx")


# 12.1.2 和讯研报网表格获取

# 它可能会弹出一个Warning警告，警告不是报错，不用在意
import pandas as pd
from selenium import webdriver
import re
# 设置Selenium的无界面模式
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(options=chrome_options)

data_all = pd.DataFrame()  # 创建一个空列表用来汇总所有表格信息
for pg in range(1, 2):  # 可以将页码调大，比如2019-04-30该天，网上一共有176页，这里可以将这个2改成176
    url = 'http://yanbao.stock.hexun.com/ybsj5_' + str(pg) + '.shtml'
    browser.get(url)  # 通过Selenium访问网站
    data = browser.page_source  # 获取网页源代码
    table = pd.read_html(data)[0]  # 通过pandas库提取表格

    # 添加股票代码信息
    p_code = '<a href="yb_(.*?).shtml'
    code = re.findall(p_code, data)
    table['股票代码'] = code

    # 通过concat()函数纵向拼接成一个总的DataFrame
    data_all = pd.concat([data_all, table], ignore_index=True)

print(data_all)
print('分析师评级报告获取成功')
data_all.to_excel('分析师评级报告.xlsx')





