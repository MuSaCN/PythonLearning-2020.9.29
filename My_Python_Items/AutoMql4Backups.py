# Author:Zhang Yuan
import MyPackage

Mql4Path = "C:\\Users\\i2011\\AppData\\Roaming\\MetaQuotes\\Terminal\\F7DC4A11FD305E0AA6ED39F4697586F7\\MQL4"

myMql4 = MyPackage.MyClass_Mql.MyClass_MqlBackups(Mql4Path,isPrint=True)

# ---MyExperts_MQL4, MyClass_MQL4, MyIndicators_MQL4, MyScripts_MQL4的复制操作
myMql4.MqlDirCopy(myMql4.ExpertsPath,"MyExperts_MQL4")
myMql4.MqlDirCopy(myMql4.IncludePath,"MyClass_MQL4")
myMql4.MqlDirCopy(myMql4.IndicatorsPath,"MyIndicators_MQL4")
myMql4.MqlDirCopy(myMql4.ScriptsPath,"MyScripts_MQL4")

# ---Files, Logs的清理操作
myMql4.MqlDirRemove(myMql4.FilesPath,ignoreFolder=[])
myMql4.MqlDirRemove(myMql4.LogsPath,ignoreFolder=[])

# ---MQL4文件夹备份
print("------开始压缩MQL4文件夹------")
needZip = Mql4Path # 需压缩的目录
# 备份到OneDrive的Work-Python备份文件夹
OneDrive_Mql4 = myMql4.myfile.ZipDir(needZip, zipPath=myMql4.mypath.GetOneDrivePath() + "\\Work-Mql备份" , zipName=None, autoName=True)
print("MQL4压缩文件保存完成，{}".format(OneDrive_Mql4))

# ---上传到Baidu云
print("------开始上传压缩文件到Baidu云盘------")
myBaidu= MyPackage.MyClass_WebCrawler.MyClass_BaiduPan()    #百度网盘交互类
needUpload = OneDrive_Mql4
remotePath = "\\MyMql4Backups\\"
# 开始批量上传
print("{} 开始上传.".format(needUpload))
myBaidu.upload(localpath=needUpload, remotepath=remotePath, ondup="overwrite")
print("{} 上传完成.".format(needUpload))

print("全部完成！")