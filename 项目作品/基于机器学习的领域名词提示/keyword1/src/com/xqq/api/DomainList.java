package com.xqq.api;

import com.xqq.rep.Document;
import opennlp.tools.postag.POSTaggerME;

import java.util.ArrayList;

/**
 * 1.判断是否进行领域文档重新聚类（更新）
 * 2.判断文档属于哪个领域
 */
public interface DomainList {
    void updateDomainList(int k, POSTaggerME tagger);
    int  judgeDomain(Document doc, double minSim);
    void insertDocument(Document doc, double minSim, POSTaggerME tagger);
    void insertDocument(Document doc, int index, POSTaggerME tagger, boolean flag);
    void save();
    void initial(int k, ArrayList<Document> docs, POSTaggerME tagger);
    void printDomainListInfo();

}
