# # 补充知识点2：新闻爬取 - 百度新闻爬虫
import requests
import re
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}

def baidu(keyword, page):
    num = (page - 1) * 10
    url = 'https://www.baidu.com/s?tn=news&rtt=4&bsst=1&cl=2&wd=' + keyword + '&pn=' + str(num)
    res = requests.get(url, headers=headers).text
    
    p_href = '<h3 class="c-title">.*?<a href="(.*?)"'
    p_title = '<h3 class="c-title">.*?>(.*?)</a>'
    p_info = '<p class="c-author">(.*?)</p>'
    href = re.findall(p_href, res, re.S)
    title = re.findall(p_title, res, re.S)
    info = re.findall(p_info, res, re.S)

    source = []
    date = []
    for i in range(len(title)):
        title[i] = title[i].strip()
        title[i] = re.sub('<.*?>', '', title[i])
        info[i] = re.sub('<.*?>', '', info[i])
        source.append(info[i].split('&nbsp;&nbsp;')[0])  
        date.append(info[i].split('&nbsp;&nbsp;')[1])
        source[i] = source[i].strip()
        date[i] = date[i].strip()
    
    result = pd.DataFrame({'关键词': keyword, '标题': title, '网址': href, '来源': source, '日期': date})
    return result
    
import pandas as pd
df = pd.DataFrame()
    
keywords = ['华能信托', '人工智能', '科技', '体育', 'Python', '娱乐', '文化', '阿里巴巴', '腾讯', '京东']
for keyword in keywords:
    for i in range(10):
        result = baidu(keyword, i+1)
        df = df.append(result)
        print(keyword + '第' + str(i+1) + '页爬取成功')

df.to_excel('新闻_new.xlsx')

df.head()

