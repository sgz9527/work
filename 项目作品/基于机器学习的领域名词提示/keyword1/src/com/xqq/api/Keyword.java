package com.xqq.api;


import com.xqq.rep.Document;
import com.xqq.rep.DocumentList;
import opennlp.tools.postag.POSTaggerME;

import java.lang.reflect.Array;
import java.util.ArrayList;

/**
 * 1.关键词提取
 * 2.关键词搜索
 * 3.更新关键词
 * 4.打印关键词信息
 */
public interface Keyword {
    void keywordExtraction(Document doc, POSTaggerME tagger);
    void keywordExtraction(ArrayList<Document> docs, POSTaggerME tagger);
    ArrayList<String> keywordSearch(String[] predictTags, String prex);
    ArrayList<String> keywordSearch(String preContext, String prex, POSTaggerME tagger);
    void updateKeyword(Document doc, POSTaggerME tagger);
    void printKeywords();
    void printKeywords(ArrayList<String> res);
    void testKeywordSearch(POSTaggerME tagger);
}
