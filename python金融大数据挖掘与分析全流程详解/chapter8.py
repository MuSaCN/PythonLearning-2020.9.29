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
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
myWebC = MyWebCrawler.MyClass_WebCrawler()  # 综合网络爬虫类
mySQL = MyDatabase.MyClass_MySQL(connect=False)  # MySQL类
#------------------------------------------------------------

# 8.2 爬虫进阶2-爬虫利器selenium库详解
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

exe_data = r'D:\软件安装地址\360Chrome\Chrome\Application\360chrome.exe'    #浏览器根目录所在地
chrome_options = Options()
chrome_options.binary_location =exe_data

browser = webdriver.Chrome(chrome_options=chrome_options)
browser.get('http://www.baidu.com')
browser.find_element_by_id("kw").send_keys("xiaobai" + Keys.RETURN)
time.sleep(10)
driver.quit()


# 1.打开及关闭网页+网页最大化
from selenium import webdriver
browser = webdriver.Chrome()
browser.maximize_window()
browser.get("https://www.baidu.com/")
browser.quit()

# 2.xpath方法来定位元素
from selenium import webdriver
browser = webdriver.Chrome()
browser.get("https://www.baidu.com/")
browser.find_element_by_xpath('//*[@id="kw"]').send_keys('python')
browser.find_element_by_xpath('//*[@id="su"]').click()

# 3.css_selector方法来定位元素
from selenium import webdriver
browser = webdriver.Chrome()
browser.get("https://www.baidu.com/")
browser.find_element_by_css_selector('#kw').send_keys('python')
browser.find_element_by_css_selector('#su').click()

# 4.browser.page_source方法来获取模拟键盘鼠标点击，百度搜索python后的网页源代码
from selenium import webdriver
import time
browser = webdriver.Chrome()
browser.get("https://www.baidu.com/")
browser.find_element_by_xpath('//*[@id="kw"]').send_keys('python')
browser.find_element_by_xpath('//*[@id="su"]').click()
time.sleep(3)  # 因为是点击按钮后跳转，所以最好休息3秒钟再进行源代码获取,如果是直接访问网站，则通常不需要等待。
data = browser.page_source
print(data)

# 5.browser.page_source方法来获取新浪财经股票信息
from selenium import webdriver
browser = webdriver.Chrome()
browser.get("http://finance.sina.com.cn/realstock/company/sh000001/nc.shtml")
data = browser.page_source
print(data)

# 6.Chrome Headless无界面浏览器设置
from selenium import webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(chrome_options=chrome_options)
browser.get("http://finance.sina.com.cn/realstock/company/sh000001/nc.shtml")
data = browser.page_source
print(data)





