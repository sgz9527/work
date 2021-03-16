package com.xqq.api;

import com.xqq.rep.DocumentList;

import java.util.ArrayList;

/**
 * 文本内容转向量接口，需要实现三个方法：1.对文档列表进行编码（转为数字向量）2.保存编码模型参数 3.从文件加载保存的编码模型参数
 */

public interface Encoder {

	public void encode(DocumentList documents);
	public void saveModel();
	public void loadModel();
	public void updateModel();
}
