# Author:Zhang Yuan
""
# ipython或cmd可直接运行
# 检查安装哪些第三方库和版本
'pip list'
# 生成到指定目录
'pip freeze > "C:\\Users\\i2011\\OneDrive\\Files_Document\\site-packages_record.txt"'
# 根据指定目录安装第三方库
'pip install -r "C:\\Users\\i2011\\OneDrive\\Files_Document\\site-packages_record.txt"'

# ---cmd方式自动执行
# 批量生成
'''
@echo off
pip list
pip freeze > "C:\\Users\\i2011\\OneDrive\\Files_Document\\site-packages_record.txt"
taskkill /f /im cmd.exe
exit
'''
# 批量安装
'''
@echo off
pip list
pip install -r "C:\\Users\\i2011\\OneDrive\\Files_Document\\site-packages_record.txt"
taskkill /f /im cmd.exe
exit
'''
