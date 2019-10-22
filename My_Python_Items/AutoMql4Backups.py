# Author:Zhang Yuan
import MyPackage
import os
__mypath__ = MyPackage.MyClass_Path.MyClass_Path()  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()      #文件操作类

# ---客户端文件夹
terminalPath="C:\\Users\\i2011\\AppData\\Roaming\\MetaQuotes\\Terminal\\F7DC4A11FD305E0AA6ED39F4697586F7"
# ---定位MQL4文件夹
MQL4Path = terminalPath+"\\MQL4"
# ---需要操作的文件夹
ExpertsPath = MQL4Path+"\\Experts"
FilesPath = MQL4Path + "\\Files"
IncludePath = MQL4Path + "\\Include"
IndicatorsPath = MQL4Path + "\\Indicators"
LogsPath = MQL4Path + "\\Logs"
ScriptsPath = MQL4Path + "\\Scripts"
print("------客户端目录：",terminalPath,
      "------MQL4文件夹目录：",MQL4Path,
      "------需要操作的文件夹：",
      ExpertsPath,FilesPath,IncludePath,IndicatorsPath,LogsPath,ScriptsPath,sep="\n")

# ---定义Mql4目录的复制函数
def Mql4DirCopy(Path,subPath):
    print("------{}文件夹开始备份------".format(subPath))
    print("{0}文件夹进行备份：{1} - 副本".format(subPath,subPath))
    Source = Path + "\\" + subPath
    Destination = Path + "\\" + subPath + " - 副本"
    myfile.copyDirOrFile(source=Source, destination=Destination,DirRemove=True)
    print("{} - 副本 备份完成".format(subPath))
# ---My_Experts, My_Include, My_Indicators, My_Scripts操作
Mql4DirCopy(ExpertsPath,"MyExperts_MQL4")
Mql4DirCopy(IncludePath,"MyClass_MQL4")
Mql4DirCopy(IndicatorsPath,"MyIndicators_MQL4")
Mql4DirCopy(ScriptsPath,"MyScripts_MQL4")

# ---定义Mql4目录的清理函数
def Mql4DirRemove(Path, ignoreFolder):
    name = os.path.split(Path)[1]
    print("------开始清理{}文件夹------".format(name))
    print("清空{0}文件夹：忽略里面 {1} 子文件夹".format(name,','.join(ignoreFolder)))
    myfile.removeDirOrFile(Path, onlyContent=True, ignoreFolder=ignoreFolder, ProtectiveCheck=False)
    print("{}文件夹清理完毕".format(name))
# ---Files, Logs操作
Mql4DirRemove(FilesPath,ignoreFolder=[])
Mql4DirRemove(LogsPath,ignoreFolder=[])

# ---MQL4文件夹备份
print("------开始压缩MQL4文件夹------")
needZip = MQL4Path # 需压缩的目录
# 备份到OneDrive的Work-Python备份文件夹
OneDrive_Mql4 = myfile.ZipDir(needZip, zipPath=__mypath__.GetOneDrivePath() + "\\Work-Mql备份" , zipName=None, autoName=True)
print("MQL4压缩文件保存完成，{}".format(OneDrive_Mql4))

# ---上传到Baidu云
print("------开始上传压缩文件到Baidu云盘------")
locallist = []
locallist.append(OneDrive_Mql4)
myBaidu= MyPackage.MyClass_WebCrawler.MyClass_BaiduPan()    #百度网盘交互类
remotePath = "\\MyMql4Backups\\"
# 开始批量上传
for i in range(len(locallist)):
    print("{} 开始上传.".format(locallist[i]))
    myBaidu.upload(localpath=locallist[i], remotepath=remotePath, ondup="overwrite")
    print("{} 上传完成.".format(locallist[i]))
print("全部完成！")
