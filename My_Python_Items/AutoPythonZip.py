# Author:Zhang Yuan
import MyPackage
__mypath__ = MyPackage.MyClass_Path.MyClass_Path()  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类


# ------MyPackage备份系列
print("------开始压缩MyPackage文件夹------")
MyPackage_PathList = __mypath__.GetMyPackagePath() # 需压缩的目录
#备份到OneDrive的Work-Python备份文件夹
ZipPath_MyPackage = __mypath__.GetOneDrivePath() + "\\Work-Python备份"
LocalPath_MyPackage = myfile.ZipDir(MyPackage_PathList[0], zipPath=ZipPath_MyPackage , zipName=None, autoName=True)
print("......MyPackage压缩文件保存完成，{}......".format(LocalPath_MyPackage))


# ------PycharmProjects备份系列
print("------开始压缩PycharmProjects文件夹------")
ProjectsPath = "C:\\Users\\i2011\\PycharmProjects"
# ---忽略.git文件夹，备份到OneDrive的Work-Python备份文件夹
ZipPath_PycharmProjects = __mypath__.GetOneDrivePath() + "\\Work-Python备份"
LocalPath_PycharmProjects = myfile.ZipDir(ProjectsPath, zipPath=ZipPath_PycharmProjects , zipName=None, autoName=True, ignoreFolder=".git")
print("......PycharmProjects压缩文件保存完成，{}......".format(LocalPath_PycharmProjects))


# ------上传到Baidu网盘------------------
'''先要授权 在Terminal中输入 bypy info ，将会出现一个提示，访问一个授权网址，在网址中输入用户名和密码，并把生成的授权码复制到命令行中。按照提示完成授权，完成了授权Python代码才能和你的百度云盘进行通信。'''
print("------开始上传压缩文件到Baidu云盘------")
locallist = []
locallist.append(LocalPath_MyPackage)
locallist.append(LocalPath_PycharmProjects)
# ---
myBaidu= MyPackage.MyClass_WebCrawler.MyClass_BaiduPan()      #百度网盘交互类
remotePath = "\\MyPythonBackups\\"
# ---开始批量上传
for i in len(locallist):
    myBaidu.upload(localpath=locallist[i], remotepath=remotePath, ondup="overwrite")
    print("{} 上传完成.".format(locallist[i]))
print("全部完成！")

