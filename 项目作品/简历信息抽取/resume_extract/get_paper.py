import requests
from bs4 import BeautifulSoup
import time

firsturl = "http://kns.cnki.net/kns/request/SearchHandler.ashx"

wantsearch = input()


def ToUtf(string):
    return string.encode('utf8')


times = time.strftime('%a %b %d %Y %H:%M:%S') + ' GMT+0800 (中国标准时间)'
headers2 = {'action': '',
            'ua': '1.11',
            'isinEn': '1',
            'PageName': 'ASP.brief_default_result_aspx',
            'DbPrefix': 'SCDB',
            'DbCatalog': '中国学术文献网络出版总库',
            'ConfigFile': 'SCDBINDEX.xml',
            'db_opt': 'CJFQ,CDFD,CMFD,CPFD,IPFD,CCND,CCJD',
            'txt_1_sel': 'SU$%=|',
            'txt_1_value1': wantsearch,
            'txt_1_special1': '%',
            'his': '0',
            'parentdb': 'SCDB',
            '__': times}
headers = {'Connection': 'Keep-Alive',
           'Accept': 'text/html,*/*',
           'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36',
           'Referer': "http://kns.cnki.net/kns/brief/default_result.aspx",
           'Cookie': 'Ecp_ClientId=5190502112900625483; Ecp_IpLoginFail=190502223.99.218.244; RsPerPage=20; cnkiUserKey=4bdba1f3-39f5-cd9f-7c1a-14309a1a6120; ASP.NET_SessionId=vydvcldogtjjv1ptz2ea5iym; SID_kns=123113; SID_klogin=125144; Hm_lvt_bfc6c23974fbad0bbfed25f88a973fb0=1556767932,1556851125; Hm_lpvt_bfc6c23974fbad0bbfed25f88a973fb0=1556851125; KNS_SortType=; SID_crrs=125133; _pk_ref=%5B%22%22%2C%22%22%2C1556851140%2C%22http%3A%2F%2Fwww.cnki.net%2F%22%5D; _pk_ses=*; SID_krsnew=125133',
           }
header = {'Connection': 'Keep-Alive',
          'Accept': 'text/html,*/*',
          'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36',
          'Referer': "http://kns.cnki.net/kns/brief/default_result.aspx?code=SCDB",
          'Cookie': 'Ecp_ClientId=5190502112900625483; Ecp_IpLoginFail=190502223.99.218.244; RsPerPage=20; cnkiUserKey=4bdba1f3-39f5-cd9f-7c1a-14309a1a6120; ASP.NET_SessionId=vydvcldogtjjv1ptz2ea5iym; SID_kns=123113; SID_klogin=125144; Hm_lvt_bfc6c23974fbad0bbfed25f88a973fb0=1556767932,1556851125; Hm_lpvt_bfc6c23974fbad0bbfed25f88a973fb0=1556851125; KNS_SortType=; SID_crrs=125133; _pk_ref=%5B%22%22%2C%22%22%2C1556851140%2C%22http%3A%2F%2Fwww.cnki.net%2F%22%5D; _pk_ses=*; SID_krsnew=125133',
          }
firstmagess = requests.get(firsturl, headers=headers, data=headers2).text
secondurl = "http://kns.cnki.net/kns/brief/brief.aspx?pagename=" + str(firstmagess) + "&S=1&sorttype="
thirdurl = 'http://kns.cnki.net/kns/brief/brief.aspx?curpage=1&RecordsPerPage=20&QueryID=6&ID=&turnpage=1&tpagemode=L&dbPrefix=SCDB&Fields=&DisplayMode=listmode&PageName=ASP.brief_default_result_aspx&isinEn=1&'
secondhtml = requests.get(thirdurl, headers=header).text
# print(secondhtml)
secondhtml = BeautifulSoup(secondhtml, "lxml")
html_title = secondhtml.select("td a[target='_blank']")
html_writer = secondhtml.select("td[class='author_flag'] a")
for i in html_title:
    print(i.text)
for i in html_writer:
    print(i.text)
