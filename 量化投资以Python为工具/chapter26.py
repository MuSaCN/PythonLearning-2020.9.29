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
myBaidu= MyPackage.MyClass_WebCrawler.MyClass_BaiduPan()      #百度网盘交互类
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\026"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

sh = pd.read_csv(Path+'\\sh50p.csv', index_col='Trddt')
sh.index = pd.to_datetime(sh.index)
formStart = '2014-01-01'
formEnd = '2015-01-01'
shform = sh[formStart:formEnd]
# 中国银行和浦发银行
PAf = shform['601988']
PBf = shform['600000']
pairf = pd.concat([PAf, PBf], axis=1)
# ---
PAflog = np.log(PAf)
retA = PAflog.diff()[1:]
PBflog = np.log(PBf)
retB = PBflog.diff()[1:]
# ---
data = pd.concat((PAflog,PBflog),1)
data.columns = ["PAf","PBf"]
model = myDA.Cointegration(var=["PBf","PAf"],data=data)
tradStart = '2015-01-01'
tradEnd = '2015-06-30'
PAt = sh.loc[tradStart:tradEnd, '601988']
PBt = sh.loc[tradStart:tradEnd, '600000']
pairt = pd.concat([PAt, PBt], axis=1)
# ---
myDA.PairTrading(pairf,pairt,isPrice=True,method="SSD",width=1.2)



model.summary()
alpha = model.params[0]
beta = model.params[1]

spreadf = PBflog - beta * PAflog - alpha
mu = np.mean(spreadf)
sd = np.std(spreadf)

# 形成期
CoSpreadT = np.log(PBt) - beta * np.log(PAt) - alpha
CoSpreadT.describe()

CoSpreadT.plot()
plt.title('交易期价差序列(协整配对)')
plt.axhline(y=mu, color='black')
plt.axhline(y=mu + 1.2 * sd, color='green')
plt.axhline(y=mu - 1.2 * sd, color='green')
plt.show()



import re
import pandas as pd
import numpy as np
from arch.unitroot import ADF
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


class PairTrading:
    def SSD(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        returnX = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
        returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
        standardX = (returnX + 1).cumprod()
        standardY = (returnY + 1).cumprod()
        SSD = np.sum((standardY - standardX) ** 2)
        return (SSD)

    def SSDSpread(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        retx = priceX.diff()[1:]
        rety = priceY.diff()[1:]
        standardX = (1 + retx).cumprod()
        standardY = (1 + rety).cumprod()
        spread = standardY - standardX
        return (spread)

    def cointegration(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        results = sm.OLS(priceY, sm.add_constant(priceX)).fit()
        resid = results.resid
        adfSpread = ADF(resid)
        if adfSpread.pvalue >= 0.05:
            print('''交易价格不具有协整关系.
            P-value of ADF test: %f
            Coefficients of regression:
            Intercept: %f
            Beta: %f
             ''' % (adfSpread.pvalue, results.params[0], results.params[1]))
            return (None)
        else:
            print('''交易价格具有协整关系.
            P-value of ADF test: %f
            Coefficients of regression:
            Intercept: %f
            Beta: %f
             ''' % (adfSpread.pvalue, results.params[0], results.params[1]))
            return (results.params[0], results.params[1])

    def CointegrationSpread(self, priceX, priceY, formPeriod, tradePeriod):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        if not (re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', formPeriod)
                or re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', tradePeriod)):
            print('形成期或交易期格式错误.')
        formX = priceX[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        formY = priceY[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        coefficients = self.cointegration(formX, formY)
        if coefficients is None:
            print('未形成协整关系,无法配对.')
        else:
            spread = (np.log(priceY[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]])
                      - coefficients[0] - coefficients[1] * np.log(
                        priceX[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]))
            return (spread)

    def calBound(self, priceX, priceY, method, formPeriod, width=1.5):
        if not (re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', formPeriod)
                or re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', tradePeriod)):
            print('形成期或交易期格式错误.')
        if method == 'SSD':
            spread = self.SSDSpread(priceX[formPeriod.split(':')[0]:formPeriod.split(':')[1]],
                                    priceY[formPeriod.split(':')[0]:formPeriod.split(':')[1]])
            mu = np.mean(spread)
            sd = np.std(spread)
            UpperBound = mu + width * sd
            LowerBound = mu - width * sd
            return (UpperBound, LowerBound)
        elif method == 'Cointegration':
            spread = self.CointegrationSpread(priceX, priceY, formPeriod, formPeriod)
            mu = np.mean(spread)
            sd = np.std(spread)
            UpperBound = mu + width * sd
            LowerBound = mu - width * sd
            return (UpperBound, LowerBound)
        else:
            print('不存在该方法. 请选择"SSD"或是"Cointegration".')


sh = pd.read_csv('sh50p.csv', index_col='Trddt')
sh.index = pd.to_datetime(sh.index)

formPeriod = '2014-01-01:2015-01-01'
tradePeriod = '2015-01-01:2015-06-30'

priceA = sh['601988']
priceB = sh['600000']
priceAf = priceA[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
priceBf = priceB[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
priceAt = priceA[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]
priceBt = priceB[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]

pt = PairTrading()
SSD = pt.SSD(priceAf, priceBf)
SSD

SSDspread = pt.SSDSpread(priceAf, priceBf)
SSDspread.describe()
SSDspread.head()

coefficients = pt.cointegration(priceAf, priceBf)
coefficients

CoSpreadF = pt.CointegrationSpread(priceA, priceB, formPeriod, formPeriod)
CoSpreadF.head()

CoSpreadTr = pt.CointegrationSpread(priceA, priceB, formPeriod, tradePeriod)
CoSpreadTr.describe()

bound = pt.calBound(priceA, priceB, 'Cointegration', formPeriod, width=1.2)
bound

# 配对交易实测
# 提取形成期数据
formStart = '2014-01-01'
formEnd = '2015-01-01'
PA = sh['601988']
PB = sh['600000']

PAf = PA[formStart:formEnd]
PBf = PB[formStart:formEnd]

# 形成期协整关系检验
# 一阶单整检验
log_PAf = np.log(PAf)
adfA = ADF(log_PAf)
print(adfA.summary().as_text())
adfAd = ADF(log_PAf.diff()[1:])
print(adfAd.summary().as_text())

log_PBf = np.log(PBf)
adfB = ADF(log_PBf)
print(adfB.summary().as_text())
adfBd = ADF(log_PBf.diff()[1:])
print(adfBd.summary().as_text())

# 协整关系检验
model = sm.OLS(log_PBf, sm.add_constant(log_PAf)).fit()
model.summary()

alpha = model.params[0]
alpha
beta = model.params[1]
beta

# 残差单位根检验
spreadf = log_PBf - beta * log_PAf - alpha
adfSpread = ADF(spreadf)

print(adfSpread.summary().as_text())

mu = np.mean(spreadf)
sd = np.std(spreadf)

# 设定交易期

tradeStart = '2015-01-01'
tradeEnd = '2015-06-30'

PAt = PA[tradeStart:tradeEnd]
PBt = PB[tradeStart:tradeEnd]

CoSpreadT = np.log(PBt) - beta * np.log(PAt) - alpha

CoSpreadT.describe()

CoSpreadT.plot()
plt.title('交易期价差序列(协整配对)')
plt.axhline(y=mu, color='black')
plt.axhline(y=mu + 0.2 * sd, color='blue', ls='-', lw=2)
plt.axhline(y=mu - 0.2 * sd, color='blue', ls='-', lw=2)
plt.axhline(y=mu + 1.5 * sd, color='green', ls='--', lw=2.5)
plt.axhline(y=mu - 1.5 * sd, color='green', ls='--', lw=2.5)
plt.axhline(y=mu + 2.5 * sd, color='red', ls='-.', lw=3)
plt.axhline(y=mu - 2.5 * sd, color='red', ls='-.', lw=3)

level = (
float('-inf'), mu - 2.5 * sd, mu - 1.5 * sd, mu - 0.2 * sd, mu + 0.2 * sd, mu + 1.5 * sd, mu + 2.5 * sd, float('inf'))

prcLevel = pd.cut(CoSpreadT, level, labels=False) - 3

prcLevel.head()


def TradeSig(prcLevel):
    n = len(prcLevel)
    signal = np.zeros(n)
    for i in range(1, n):
        if prcLevel[i - 1] == 1 and prcLevel[i] == 2:
            signal[i] = -2
        elif prcLevel[i - 1] == 1 and prcLevel[i] == 0:
            signal[i] = 2
        elif prcLevel[i - 1] == 2 and prcLevel[i] == 3:
            signal[i] = 3
        elif prcLevel[i - 1] == -1 and prcLevel[i] == -2:
            signal[i] = 1
        elif prcLevel[i - 1] == -1 and prcLevel[i] == 0:
            signal[i] = -1
        elif prcLevel[i - 1] == -2 and prcLevel[i] == -3:
            signal[i] = -3
    return (signal)


signal = TradeSig(prcLevel)

position = [signal[0]]
ns = len(signal)

for i in range(1, ns):
    position.append(position[-1])
    if signal[i] == 1:
        position[i] = 1
    elif signal[i] == -2:
        position[i] = -1
    elif signal[i] == -1 and position[i - 1] == 1:
        position[i] = 0
    elif signal[i] == 2 and position[i - 1] == -1:
        position[i] = 0
    elif signal[i] == 3:
        position[i] = 0
    elif signal[i] == -3:
        position[i] = 0

position = pd.Series(position, index=CoSpreadT.index)

position.tail()


def TradeSim(priceX, priceY, position):
    n = len(position)
    size = 1000
    shareY = size * position
    shareX = [(-beta) * shareY[0] * priceY[0] / priceX[0]]
    cash = [2000]
    for i in range(1, n):
        shareX.append(shareX[i - 1])
        cash.append(cash[i - 1])
        if position[i - 1] == 0 and position[i] == 1:
            shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
            cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
        elif position[i - 1] == 0 and position[i] == -1:
            shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
            cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
        elif position[i - 1] == 1 and position[i] == 0:
            shareX[i] = 0
            cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
        elif position[i - 1] == -1 and position[i] == 0:
            shareX[i] = 0
            cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
    cash = pd.Series(cash, index=position.index)
    shareY = pd.Series(shareY, index=position.index)
    shareX = pd.Series(shareX, index=position.index)
    asset = cash + shareY * priceY + shareX * priceX
    account = pd.DataFrame({'Position': position, 'ShareY': shareY, 'ShareX': shareX, 'Cash': cash, 'Asset': asset})
    return (account)


account = TradeSim(PAt, PBt, position)
account.tail()

account.iloc[:, [0, 1, 4]].plot(style=['--', '-', ':'])
plt.title('配对交易账户')












