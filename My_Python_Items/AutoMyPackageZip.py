# Author:Zhang Yuan
from MyPackage.MyPath import MyClass_Path
from MyPackage.MyFile import MyClass_File

__mypath__ = MyClass_Path()  #路径类
myfile = MyClass_File()  #文件操作类

MyPackage_PathList = __mypath__.GetMyPackagePath()
#MyPackage_PathList = ["C:\\Users\\i2011\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\MyPackage"]

# ---备份到桌面
myfile.ZipDir(MyPackage_PathList[0], zipPath="Desktop" , zipName=None, autoName=True)

# ---备份到OneDrive的Work-Python备份文件夹
ZipPath = __mypath__.GetOneDrivePath() + "\\Work-Python备份"
myfile.ZipDir(MyPackage_PathList[0], zipPath=ZipPath , zipName=None, autoName=True)



