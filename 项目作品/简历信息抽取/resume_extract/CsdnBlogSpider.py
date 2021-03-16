# encoding:utf-8
__author__ = 'Sun'

import re
import urllib.request
import urllib
import queue
import threading
import os
import random
from bs4 import BeautifulSoup
import jieba
from special_string import USER_DIC_PATH
from create_data import stopwordslist

import time


queue = queue.Queue()
visited = set()
cnt = 0
count = 0
black_list = {
	'html', 'head', 'meta', 'script', 'style', 'body', 'div', 'p', 'span'
}


class CsdnBlogSpider(threading.Thread):

	def __init__(self, queue, opener, blog_name):
		threading.Thread.__init__(self)
		self.queue = queue
		self.opener = opener
		self.blog_name = blog_name
		self.lock = threading.Lock()
		self.stop_list = stopwordslist()
		jieba.load_userdict(USER_DIC_PATH)

	def save_data(self, data, filename):
		blog_path = './blog'
		if not os.path.exists('blog'):
			blog_path = os.path.join(os.path.abspath('.'),'blog')
			os.mkdir(blog_path)
		filenames = os.listdir(blog_path)
		filenames = [name.split('.')[0] for name in filenames]
		str_time = time.strftime('%H-%M-%S', time.localtime())
		filename += str_time
		while filename in filenames:
			filename += str(random.randint(0, 9))
		try:
			# print(filename)
			fout = open('./blog/' + filename + '.txt', 'w', encoding='utf-8')
			soup = BeautifulSoup(data, 'html.parser')
			text = soup.findAll(text=True)  # 找到所有文本内容
			for t in text:
				if t.parent.name not in black_list:
					t = re.sub(r'[^\u4E00-\u9FD5\d\w\]\[://@.]+', '', t)
					if len(t) > 8:
						tex_ls = jieba.lcut(t)
						tex_ls = [item for item in tex_ls if item not in self.stop_list]
						txt = ' '.join(tex_ls)
						fout.write(txt)
						fout.write('\n')
		except IOError as e:
			print(e)
		# finally:
		# 	fout.close()

	def find_title(self,data):
		data = data.decode('utf-8')
		begin = data.find(r'<title') + 7
		end = data.find('\r\n',begin)
		title = data[begin:end]
		return title

	def run(self):
		global cnt
		global visited
		while True:
			url = self.queue.get()
			self.lock.acquire()
			cnt += 1
			print('已经抓取：' + str(cnt-1) + '正在抓取---->' + url)
			self.lock.release()
			try:
				res = self.opener.open(url, timeout=1000)
			except Exception as e:
				if hasattr(e, 'reason'):
					print('reason:', e.reason)
				elif hasattr(e, 'code'):
					print('error code:', e.code)
				cnt -= 1
				self.queue.task_done()
				continue
			else:
				data = res.read()
			title = self.find_title(data)
			global count
			self.save_data(data,str(count))
			count+=1

			data = data.decode('utf-8')
			blog_urls = re.compile('/' + self.blog_name + '/article/details/' + '\d*')
			for url in blog_urls.findall(data):
				url = 'http://blog.csdn.net' + url
				if url not in visited:
					self.queue.put(url)
					visited |= {url}
					# print('加入队列---》' + url)
			self.queue.task_done()

def init(name, number=10):
	global cnt
	global visited
	# blog_name = input('输入博客名称:')
	# thread_num = input('输入启动线程数:')
	blog_name = name.lower()
	th_num = int(number)
	url = 'http://blog.csdn.net/' + blog_name + '/'
	opener = urllib.request.build_opener(urllib.request.HTTPHandler)
	headers = [
		('User-Agent', 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko')
	]
	urllib.request.install_opener(opener)
	opener.addheaders = headers

	queue.put(url)
	visited |= {url}
	cnt = 0

	for i in range(th_num):
		t = CsdnBlogSpider(queue,opener,blog_name)
		t.setDaemon(True)
		t.start()
	queue.join()
	print('--------end!!!-----')
	print('共抓取:' + str(cnt))

if __name__ == '__main__':
	init()


