# Author:Zhang Yuan
import MyPackage
import os
__mypath__ = MyPackage.MyClass_Path.MyClass_Path()  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()      #文件操作类

# ---客户端文件夹
terminalPath="C:\\Users\\i2011\\AppData\\Roaming\\MetaQuotes\\Terminal\\6E8A5B613BD795EE57C550F7EF90598D"
# ---定位MQL5文件夹
MQL5Path = terminalPath+"\\MQL5"
# ---需要操作的文件夹
ExpertsPath = MQL5Path+"\\Experts"
FilesPath = MQL5Path + "\\Files"
IncludePath = MQL5Path + "\\Include"
IndicatorsPath = MQL5Path + "\\Indicators"
LogsPath = MQL5Path + "\\Logs"
ScriptsPath = MQL5Path + "\\Scripts"
print("------客户端目录：",terminalPath,
      "------MQL5文件夹目录：",MQL5Path,
      "------需要操作的文件夹：",
      ExpertsPath,FilesPath,IncludePath,IndicatorsPath,LogsPath,ScriptsPath,sep="\n")

# ---定义Mql5目录的复制函数
def Mql5DirCopy(Path,subPath):
    print("------{}文件夹开始备份------".format(subPath))
    print("{0}文件夹进行备份：{1} - 副本".format(subPath,subPath))
    Source = Path + "\\" + subPath
    Destination = Path + "\\" + subPath + " - 副本"
    myfile.copyDirOrFile(source=Source, destination=Destination,DirRemove=True)
    print("{} - 副本 备份完成".format(subPath))
# ---My_Experts, My_Include, My_Indicators, My_Scripts操作
Mql5DirCopy(ExpertsPath,"My_Experts")
Mql5DirCopy(IncludePath,"My_Include")
Mql5DirCopy(IndicatorsPath,"My_Indicators")
Mql5DirCopy(ScriptsPath,"My_Scripts")

# ---定义Mql5目录的清理函数
def Mql5DirRemove(Path, ignoreFolder):
    name = os.path.split(Path)[1]
    print("------开始清理{}文件夹------".format(name))
    print("清空{0}文件夹：忽略里面 {1} 子文件夹".format(name,','.join(ignoreFolder)))
    myfile.removeDirOrFile(Path, onlyContent=True, ignoreFolder=ignoreFolder, ProtectiveCheck=False)
    print("{}文件夹清理完毕".format(name))
# ---Files, Logs操作
Mql5DirRemove(FilesPath,ignoreFolder=["SPSS"])
Mql5DirRemove(LogsPath,ignoreFolder=[])

# ---MQL5文件夹备份
print("------开始压缩MQL5文件夹------")
needZip = MQL5Path # 需压缩的目录
# 备份到OneDrive的Work-Python备份文件夹
OneDrive_Mql5 = myfile.ZipDir(needZip, zipPath=__mypath__.GetOneDrivePath() + "\\Work-Mql备份" , zipName=None, autoName=True)
print("MQL5压缩文件保存完成，{}".format(OneDrive_Mql5))

# ---上传到Baidu云
print("------开始上传压缩文件到Baidu云盘------")
locallist = []
locallist.append(OneDrive_Mql5)
myBaidu= MyPackage.MyClass_WebCrawler.MyClass_BaiduPan()    #百度网盘交互类
remotePath = "\\MyMql5Backups\\"
# 开始批量上传
for i in range(len(locallist)):
    print("{} 开始上传.".format(locallist[i]))
    myBaidu.upload(localpath=locallist[i], remotepath=remotePath, ondup="overwrite")
    print("{} 上传完成.".format(locallist[i]))
print("全部完成！")

