# Author:Zhang Yuan
import MyPackage
__mypath__ = MyPackage.MyClass_Path.MyClass_Path()  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类


# ------MyPackage备份系列
print("------开始压缩MyPackage文件夹------")
MyPackage_PathList = __mypath__.GetMyPackagePath() # 需压缩的目录
# 备份到OneDrive的Work-Python备份文件夹
OneDrive_MyPackage = myfile.ZipDir(MyPackage_PathList[0], zipPath=__mypath__.GetOneDrivePath() + "\\Work-Python备份" , zipName=None, autoName=True)
print("......MyPackage压缩文件保存完成，{}......".format(OneDrive_MyPackage))
# 备份到桌面(用于上传)
# Desktop_MyPackage = myfile.ZipDir(MyPackage_PathList[0], zipPath="Desktop", zipName=None, autoName=True)
# print("......MyPackage压缩文件保存完成，{}......".format(Desktop_MyPackage))


# ------PycharmProjects备份系列
print("------开始压缩PycharmProjects文件夹------")
ProjectsPath = "C:\\Users\\i2011\\PycharmProjects"
# 忽略.git文件夹，备份到OneDrive的Work-Python备份文件夹
OneDrive_PycharmProjects = myfile.ZipDir(ProjectsPath, zipPath=__mypath__.GetOneDrivePath() + "\\Work-Python备份" , zipName=None, autoName=True, ignoreFolder=".git")
print("......PycharmProjects压缩文件保存完成，{}......".format(OneDrive_PycharmProjects))
# 忽略.git文件夹，备份到桌面(用于上传)
# Desktop_PycharmProjects = myfile.ZipDir(ProjectsPath, zipPath="Desktop" , zipName=None, autoName=True, ignoreFolder=".git")
# print("......PycharmProjects压缩文件保存完成，{}......".format(Desktop_PycharmProjects))


# ------上传到Baidu网盘------------------
'''先要授权 在Terminal中输入 bypy info ，将会出现一个提示，访问一个授权网址，在网址中输入用户名和密码，并把生成的授权码复制到命令行中。按照提示完成授权，完成了授权Python代码才能和你的百度云盘进行通信。'''
print("------开始上传压缩文件到Baidu云盘------")
locallist = []
# locallist.append(Desktop_MyPackage)
# locallist.append(Desktop_PycharmProjects)
locallist.append(OneDrive_MyPackage)
locallist.append(OneDrive_PycharmProjects)

# ---引入类和设定网盘目录
myBaidu= MyPackage.MyClass_WebCrawler.MyClass_BaiduPan()      #百度网盘交互类
remotePath = "\\MyPythonBackups\\"
# ---开始批量上传
for i in range(len(locallist)):
    print("{} 开始上传.".format(locallist[i]))
    myBaidu.upload(localpath=locallist[i], remotepath=remotePath, ondup="overwrite")
    print("{} 上传完成.".format(locallist[i]))
print("全部完成！")
