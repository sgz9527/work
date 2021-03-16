# -*- coding: utf-8 -*-
"""
本模块将原简历pdf或docx文档中的文字提取出来
"""
from os.path import isfile, join
import os
from special_string import *
"""
from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
"""
import subprocess


class CPdf2TxtManager():

  def __init__(self):
    """
    constructor
    """
    self.ef = ROOT_PATH + '\\xpdf\\pdftotext.exe'
    self.cfg = ROOT_PATH + '\\xpdf\\xpdfrc'
    # self.ef = "./pdftotext.exe"
    # self.cfg = "./xpdfrc"

  def convert_pdf_to_txt(self,file):
      cmd = [self.ef, '-layout','-cfg', self.cfg, '-raw', file, '-']
      txt = subprocess.check_output(cmd)
      import chardet
      code_info = chardet.detect(txt)  # 编码信息,{'encoding':...,'confidence':..,'language':...}
      if code_info['encoding'] == 'utf-8' or code_info['encoding'] == 'UTF-8':
        return txt.decode('UTF-8')
        # return txt.decode('gb2312','ignore').encode('utf-8').decode('utf-8')
      return txt.decode('utf-8','ignore')

  def changePdfToText(self, filePath):
    """
    file = open(filePath, 'rb')  # 以二进制读模式打开
    # 用文件对象来创建一个pdf文档分析器
    praser = PDFParser(file)
    # 创建一个PDF文档
    doc = PDFDocument()
    # 连接分析器 与文档对象
    praser.set_document(doc)
    doc.set_parser(praser)
    # 提供初始化密码
    # 如果没有密码 就创建一个空的字符串
    doc.initialize()
    # 检测文档是否提供txt转换，不提供就忽略
    if not doc.is_extractable:
      raise PDFTextExtractionNotAllowed
    # 创建PDf 资源管理器 来管理共享资源
    rsrcmgr = PDFResourceManager()
    # 创建一个PDF设备对象
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # 创建一个PDF解释器对象
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pdfStr = ''
    # 循环遍历列表，每次处理一个page的内容
    for page in doc.get_pages(): # doc.get_pages() 获取page列表
      interpreter.process_page(page)
      # 接受该页面的LTPage对象
      layout = device.get_result()
      # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等 想要获取文本就获得对象的text属性，
      for x in layout:
        if (isinstance(x, LTTextBoxHorizontal)):
          pdfStr = pdfStr + x.get_text() + '\n'
    file.close()
    """
    return ""


if __name__ == '__main__':
  """
   解析pdf 文本
  """
  TRAIN_PATH = MATERIAL_TRAIN_PATH
  pdf2TxtManager = CPdf2TxtManager()
  file_names = os.listdir(TRAIN_PATH)
  for item in file_names:
    if isfile(join(TRAIN_PATH,item)) and item.endswith('.pdf'):
      print(pdf2TxtManager.convert_pdf_to_txt(join(TRAIN_PATH, item)))
      print('')
      print('')
      print('')
      print('')
      print('')
      print('')
      print('')
      print('')
      print('')
      print('')
      print('')