# Author:Zhang Yuan
# ------------------使用说明------------------------------------------------
# 文件命名规则：
#   logo原始文件命名规则：abc_logo.png --> abc_logo_*_DEMO.png / abc_logo_*_Paid.png
#   screenshot原始文件命名规则：abc.png --> abc_640_480.png
# logo图片添加文字：
#   IndicatorName = [*] / myImage.textOnImage() 为logo核心文本输入内容，每次使用都不同，需要修改。


from MyPackage.MyPath import MyClass_Path
from MyPackage.MyFile import MyClass_File
from MyPackage.MyImage import MyClass_ImageProcess

__mypath__ = MyClass_Path()  #路径类
myfile = MyClass_File()  #文件操作类
myImage = MyClass_ImageProcess()  # 图片处理类

# ---获得指定目录下的所有image文件
filepath = __mypath__.GetDesktopPath()
imagelist = myImage.search_image(filepath)

# ---处理image
for image in imagelist:
    file_in = filepath + "\\" + image
    # 必须先加载image才可以处理
    myImage.__init__(file_in)
    name = image.split(".")
    # 文件名后面标识符判定
    if myfile.fileIdentifier(image,sep="_",target="logo")== True:
        # 如果是logo文件，转成200*200
        width = 200 ; height = 200
        # 以 *_logo_200_200.* 方式命名，存入logo文件夹
        file_out = filepath + "\\logo\\" + name[0] + "." + name[1]
        # 获取修改大小后的内存
        myImage.resizeImage(width, height, file_out, save=True)
        # ！！！！！！logo处理方案！！！！！！！！！！！！！
        IndicatorName = ["AC","Alligator","AO","BWMFI","Fractals","Gator","ATR","BearsPower","BullsPower","CCI","Chaikin","DeMarker","Force","MACD","Momentum","OsMA","RSI","RSV","Stochastic","TriX","WPR","AD","MFI","OBV","Volumes","ADX","ADXWilder","AMA","Bands","DEMA","Envelopes","FrAMA","Ichimoku","MA","SAR","StdDev","TEMA","VIDyA"]  # ***每次需修改***
        for i in IndicatorName:
            # 生成logo_DEMO
            myImage.__init__(file_in = file_out, draw = True)
            file_demo = filepath + "\\logo\\" + name[0] + "_" +i + "_DEMO." + name[1]
            myImage.textOnImage(i, -1, -1, 35, file_demo, save=False)
            #myImage.textOnImage("MoreTimeFrame", -1, 120, 25, file_demo, save=False) # ***每次需修改***
            myImage.textOnImage("DEMO", -1, 158, 30, file_demo, save=True)
            # 生成logo_Paid
            myImage.__init__(file_in=file_out, draw=True)
            file_paid = filepath + "\\logo\\" + name[0] + "_" +i + "_Paid." + name[1]
            myImage.textOnImage(i, -1, -1, 35, file_paid, save=True)
            #myImage.textOnImage("MoreTimeFrame", -1, 120, 25, file_paid, save=True)  # ***每次需修改***
    else:
        # 不是logo文件，则转成640*480
        width = 640; height = 480
        # 以 *_640_480.* 方式命名，存入screenshot文件夹
        file_out = filepath + "\\screenshot\\" + name[0] + "_640_480." + name[1]
        myImage.resizeImage(width, height, file_out, save = True)




