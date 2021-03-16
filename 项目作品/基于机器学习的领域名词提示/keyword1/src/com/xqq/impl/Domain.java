package com.xqq.impl;

import com.xqq.api.Keyword;
import com.xqq.rep.Document;
import com.xqq.rep.DocumentList;
import com.xqq.rep.WordsVector;
import opennlp.tools.postag.POSTaggerME;

import java.io.Serializable;

public class Domain implements  Serializable {

    private DocumentList documentList;  // 领域文档集合
    private Keyword keyword;  // 关键词信息

    public WordsVector getCentroVec() {
        return centroVec;
    }

    public void setCentroVec(WordsVector centroVec) {
        this.centroVec = centroVec;
    }

    private WordsVector centroVec;//领域向量


    public Domain() {
    }

    public Domain(DocumentList docs, Keyword keys, WordsVector vec ) {
        this.documentList = docs;
        this.keyword = keys;
        this.centroVec = vec;
    }

    public void updateDomain(Document doc, POSTaggerME tagger){
        // 插入一份文档需要更新领域的文档集合和关键词信息
        int num = this.documentList.getDocuments().size();
        this.centroVec.multiply(num);
        this.centroVec.add(doc.getTfidf());
        this.centroVec.divide(num + 1);
        documentList.updateDocmentList(doc);
        keyword.updateKeyword(doc, tagger);


    }
    public Keyword getKeyword() {
        return keyword;
    }

    public void setKeyword(Keyword keyword) {
        this.keyword = keyword;
    }

    public DocumentList getDocumentList() {
        return documentList;
    }

    public void setDocumentList(DocumentList documentList) {
        this.documentList = documentList;
    }

    public static void main(String[] args) {
        System.out.println("main");
    }

}
