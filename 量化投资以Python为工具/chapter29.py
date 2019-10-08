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
__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\量化投资以Python为工具")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()  #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\029"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

#strategy
BOCM=pd.read_csv(Path+'\\BOCM.csv')
BOCM.index=BOCM.iloc[:,1]
BOCM.index=pd.to_datetime(BOCM.index, format='%Y-%m-%d')
BOCMclp=BOCM.Close

rsi6=myDA.rsi(BOCMclp,6)
rsi24=myDA.rsi(BOCMclp,24)

# rsi6捕捉买卖点
Signal1=pd.Series(0,index=rsi6.index)
for i in rsi6.index:
    if rsi6[i]>80:
        Signal1[i]= -1
    elif rsi6[i]<20:
        Signal1[i]= 1
    else:
        Signal1[i]= 0

# 交叉信号
Signal2=pd.Series(0,index=rsi24.index)
lagrsi6= rsi6.shift(1)
lagrsi24= rsi24.shift(1)
for i in rsi24.index:
    if (rsi6[i]>rsi24[i]) & (lagrsi6[i]<lagrsi24[i]):
        Signal2[i]=1
    elif (rsi6[i]<rsi24[i]) & (lagrsi6[i]>lagrsi24[i]):
        Signal2[i]=-1

# 信号合并
signal=Signal1+Signal2
signal[signal>=1]=1
signal[signal<=-1]=-1
signal=signal.dropna()

# 交易：当信号出现后，下一期交易
tradSig=signal.shift(1)

ret=BOCMclp/BOCMclp.shift(1)-1
ret.head()

ret=ret[tradSig.index] # 交易期的当期收益率
buy=tradSig[tradSig==1]
buyRet=ret[tradSig==1]*buy # 买入收益率

sell=tradSig[tradSig==-1]
sellRet=ret[tradSig==-1]*sell # 卖出收益率

tradeRet=ret*tradSig # 总交易收益率

# plot
plt.subplot(211)
plt.plot(buyRet,label="buyRet",color='g')
plt.plot(sellRet,label="sellRet",color='r',linestyle='dashed')
plt.title("RSI指标交易策略")
plt.ylabel('strategy return')
plt.legend()
plt.subplot(212)
plt.plot(ret,'b')
plt.ylabel('stock return')
plt.show()


def strat(tradeSignal,ret):
    indexDate=tradeSignal.index
    ret=ret[indexDate]
    tradeRet=ret*tradeSignal
    tradeRet[tradeRet==(-0)]=0
    winRate=len(tradeRet[tradeRet>0])/len(tradeRet[tradeRet!=0])
    meanWin=sum(tradeRet[tradeRet>0])/len(tradeRet[tradeRet>0])
    meanLoss=sum(tradeRet[tradeRet<0])/len(tradeRet[tradeRet<0])
    perform={'winRate':winRate,'meanWin':meanWin,'meanLoss': meanLoss}
    return(perform)

BuyOnly=strat(buy,ret)
SellOnly=strat(sell,ret)
Trade=strat(tradSig,ret)
Test=pd.DataFrame({"BuyOnly":BuyOnly,"SellOnly":SellOnly,"Trade":Trade})
Test

#累计收益率
cumStock=np.cumprod(1+ret)-1
cumTrade=np.cumprod(1+tradeRet)-1

plt.subplot(211)
plt.plot(cumStock)
plt.ylabel('cumStock')
plt.title('股票本身累计收益率')
plt.subplot(212)
plt.plot(cumTrade)
plt.ylabel('cumTrade')
plt.title('rsi策略累计收益率')
plt.show()

#修正策略
tradSig2=signal.shift(3)
ret2=ret[tradSig2.index]
buy2=tradSig[tradSig2==1]
buyRet2=ret2[tradSig2==1]*buy2
sell2=tradSig2[tradSig2==-1]
sellRet2=ret2[tradSig2==-1]*sell2
tradeRet2=ret2*tradSig2
BuyOnly2=strat(buy2,ret2)
SellOnly2=strat(sell2,ret2)
Trade2=strat(tradSig2,ret2)
Test2=pd.DataFrame({"BuyOnly":BuyOnly2,"SellOnly":SellOnly2,"Trade":Trade2})
Test2

cumStock2=np.cumprod(1+ret2)-1
print(cumStock2[-1])
cumTrade2=np.cumprod(1+tradeRet2)-1
print(cumTrade2[-1])

plt.subplot(211)
plt.plot(cumStock2)
plt.ylabel('cumStock2')
plt.title('股票本身累计收益率')
plt.subplot(212)
plt.plot(cumTrade2)
plt.ylabel('cumTrade2')
plt.title('修改rsi执行策略累计收益率')
plt.show()


