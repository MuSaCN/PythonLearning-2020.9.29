# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("\\DeepLearningWithPython")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
mytime = MyTime.MyClass_Time()  # 时间类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDatabase.MyClass_MySQL(connect=False)  # MySQL类
mySQLAPP = MyDatabase.MyClass_SQL_APPIntegration()  # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
myFactorD = MyQuant.MyClass_Factor_Detection()  # 因子检测类
myKeras = MyDeepLearning.MyClass_Keras()  # Keras综合类
#------------------------------------------------------------


#%%
import keras
keras.__version__

import os
original_dataset_dir = os.path.expandvars('%USERPROFILE%')+'\\.kaggle\\dogs-vs-cats'
base_dir = original_dataset_dir+'\\cats_and_dogs_small'

model = myKeras.load_model(base_dir+'\\cats_and_dogs_small_2.h5')
model.summary()  # As a reminder.

#%%
img_path = base_dir+'\\test\\cats\\cat.1700.jpg'
myKeras.plot_cnn2D_layers(model,img_path,plot_origin=True,layerslimit=8,layerlevel=2)

#%%
from keras.applications import VGG16
model = VGG16(weights='imagenet', include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
aaa = myKeras.plot_cnn2D_filter(model,layer_name,filter_index,stepscount=40)

#%%
model.summary()
for layer_name in ['block1_conv1','block2_conv1']:
    results = myKeras.plot_cnn2D_filter(model, layer_name, filter_index=None, stepscount=40)

#%%
from keras import backend as K
from keras.applications.vgg16 import VGG16
K.clear_session()
# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model = VGG16(weights='imagenet')
model.summary()

#%%
from keras.applications.vgg16 import decode_predictions

# The local path to our target image
img_path = 'creative_commons_elephant.jpg'
last_conv_layer_name = "block5_conv3"

from keras.preprocessing import image
# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img) # shape (224, 224, 3)
x = np.expand_dims(x, axis=0) # shape (1, 224, 224, 3)
# 对批量进行预处理（按通道进行颜色标准化）
from keras.applications.imagenet_utils import preprocess_input
x = preprocess_input(x)

#%%
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0]) # 预测向量解码为人类可读的格式

#%%
# 预测向量中的“非洲象”元素
african_elephant_output = model.output[:, np.argmax(preds[0])]
# block5_conv3 层的输出特征图，它是 VGG16 的最后一个卷积层
last_conv_layer = model.get_layer(last_conv_layer_name)
# “非洲象”类别相对于 block5_conv3 输出特征图的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
# 形状为 (512,) 的向量，每个元素是特定特征图通道的梯度平均大小
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# 访问刚刚定义的量：对于给定的样本图像，pooled_grads 和 block5_conv3 层的输出特征图
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# 对于两个大象的样本图像，这两个量都是 Numpy 数组
pooled_grads_value, conv_layer_output_value = iterate([x])
# 将特征图数组的每个通道乘以“这个通道对‘大象’类别的重要程度”.
for i in range(conv_layer_output_value.shape[-1]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# 得到的特征图的逐通道平均值即为类激活的热力图
heatmap = np.mean(conv_layer_output_value, axis=-1)

#%%
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

#%%
# 用 OpenCV 来生成一张图像
import cv2
# 用 cv2 加载原始图像
img = cv2.imread(img_path)
# 将热力图的大小调整为与原始图像相同
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# 将热力图转换为 RGB 格式
heatmap = np.uint8(255 * heatmap)
# 将热力图应用于原始图像
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 这里的 0.4 是热力图强度因子
superimposed_img = heatmap * 0.4 + img

myplt.set_style("defaults")
plt.figure()
plt.imshow(superimposed_img/255)
plt.grid(False)
plt.show()

# 将图像保存到硬盘
cv2.imwrite('elephant_cam.jpg', superimposed_img)





