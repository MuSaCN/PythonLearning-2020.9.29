# Author:Zhang Yuan
import MyPackage
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
      ExpertsPath,FilesPath,IncludePath,IndicatorsPath,LogsPath,ScriptsPath,
      sep="\n")

# ---My_Experts操作
print("------My_Experts文件夹开始操作------")
print("My_Experts文件夹进行备份：My_Experts - 副本")
myfile.copyDirOrFile(source=ExpertsPath+"\\My_Experts",destination=ExpertsPath+"\\My_Experts - 副本",DirRemove=True)
print("My_Experts - 副本 备份完成")
print("\n")

# ---My_Include操作
print("------My_Include文件夹开始操作------")
print("My_Include文件夹进行备份：My_Include - 副本")
myfile.copyDirOrFile(source=IncludePath+"\\My_Include",destination=IncludePath+"\\My_Include - 副本",DirRemove=True)
print("My_Include - 副本 备份完成")
print("\n")

# ---My_Indicators操作
print("------My_Indicators文件夹开始操作------")
print("My_Indicators文件夹进行备份：My_Indicators - 副本")
myfile.copyDirOrFile(source=IndicatorsPath+"\\My_Indicators",destination=IndicatorsPath+"\\My_Indicators - 副本",DirRemove=True)
print("My_Indicators - 副本 备份完成")
print("\n")

# ---My_Scripts操作
print("------My_Scripts文件夹开始操作------")
print("My_Scripts文件夹进行备份：My_Scripts - 副本")
myfile.copyDirOrFile(source=ScriptsPath+"\\My_Scripts",destination=ScriptsPath+"\\My_Scripts - 副本",DirRemove=True)
print("My_Scripts - 副本 备份完成")
print("\n")

# ---Files操作
print("------Files文件夹开始操作------")
print("清空Files文件夹：忽略里面 SPSS子文件夹")
myfile.removeDirOrFile(FilesPath, onlyContent=True,ignoreFolder=["SPSS"], ProtectiveCheck=False)
print("Files文件夹清理完毕")
print("\n")

# ---Logs操作
print("------Logs文件夹开始操作------")
print("清空Logs文件夹：")
myfile.removeDirOrFile(LogsPath, onlyContent=True,ignoreFolder=None, ProtectiveCheck=False)
print("Logs文件夹清理完毕")
print("\n")

# ---MQL5文件夹备份
print("------开始压缩MQL5文件夹------")
needZip = MQL5Path # 需压缩的目录
# 备份到OneDrive的Work-Python备份文件夹
OneDrive_Mql5 = myfile.ZipDir(needZip, zipPath=__mypath__.GetOneDrivePath() + "\\Work-Mql备份" , zipName=None, autoName=True)
print("MQL5压缩文件保存完成，{}".format(OneDrive_Mql5))
print("\n")

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

