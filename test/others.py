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
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myWebQD = MyPackage.MyClass_WebCrawler.MyClass_WebQuotesDownload()  #金融行情下载类
#------------------------------------------------------------


def decorator_maker_with_arguments(decorator_arg1, decorator_arg2):
    def my_decorator(func):
        def wrapped(function_arg1, function_arg2) :
            print (decorator_arg1, decorator_arg2,function_arg1, function_arg2)
            return func(function_arg1, function_arg2)
        return wrapped
    return my_decorator


def my_decorator(self):
    def wrapped(func) :
        def next(*args,**kwargs):
            print(*args,**kwargs)
        return next
    return wrapped


@my_decorator("self")
def decorated_function_with_arguments(self,d):
    print("OK")


decorated_function_with_arguments( "ABC","DEF")


class A:
    def __init__(self):
        global v
        v = self
    def aaa(self):
        print(123)
    def bbb(self):
        v.aaa()

