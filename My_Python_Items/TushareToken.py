# Author:Zhang Yuan

import MyPackage.MyWebCrawler.WebQuotesDownload as WebQD

myWebQD = []
for i in range(len(WebQD.TOKENLIST)):
    myWebQD.append(WebQD.MyClass_WebQuotesDownload(True, token=WebQD.TOKENLIST[i]))

for i in range(len(myWebQD)):
    try:
        myWebQD[i].pro.stk_limit(ts_code='002149.SZ', start_date='20190115', end_date='20190615')
        print("积分足够%s: "%str(i), WebQD.TOKENLIST[i])
    except:
        print("积分不够%s: "%str(i), WebQD.TOKENLIST[i])

codelist = [str(i).rjust(6,'0')+".SZ" for i in range(100)]
for i in range(len(myWebQD)):
    for j in range(len(codelist)):
        a = myWebQD[i].pro.daily(ts_code=codelist[j], start_date='20190701', end_date='20190718')
        if len(a) == 0:
            print("没有这个代码: ", codelist[j])
    print("刷完%s: "%str(i), WebQD.TOKENLIST[i])






