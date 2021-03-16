package com.xqq.impl;

import com.xqq.Constants;
import com.xqq.api.DomainList;
import com.xqq.api.Keyword;
import com.xqq.rep.*;
import opennlp.tools.postag.POSTaggerME;

import java.io.Serializable;
import java.util.ArrayList;

public class DomainListImpl implements DomainList, Serializable {
    public static final int CACHE_FILE_MAX_NUM = Constants.CACHE_FILE_MAX_NUM;
    public static final String DOMAINLIST_FILE_PATH = Constants.DOMAIN_LIST_PATH;
    private ArrayList<Domain> domainList;

    private DocumentList cacheList;

    public DocumentList getCacheList() {
        return cacheList;
    }

    public void setCacheList(DocumentList cacheList) {
        this.cacheList = cacheList;
    }

    public ArrayList<Domain> getDomainList() {
        return domainList;
    }

    public void setDomainList(ArrayList<Domain> domainList) {
        this.domainList = domainList;
    }

    /**
     * 该构造函数只会在第一次加载时被调用（在这之前还不存在domainList 模型文件)
     * @param k 初始聚类的类别个数
     */

    public DomainListImpl( int k, ArrayList<Document> docs, POSTaggerME tagger){
        this.domainList = new ArrayList<>();
        this.cacheList = new DocumentList();
        this.initial(k, docs, tagger);
    }


    /**
     *
     * @param k 更新后聚类的堆数
     */
    @Override
    public void updateDomainList(int k, POSTaggerME tagger) {
        TfidfEncoder tfidf = new TfidfEncoder(false);
        ArrayList<Document> all = new ArrayList<>();
        for(int i=0;i<domainList.size();++i){
            all.addAll(domainList.get(i).getDocumentList().getDocuments());
        }
        all.addAll(cacheList.getDocuments());
        domainList.clear();
        this.cacheList.getDocuments().clear();
        DocumentList allDocs = new DocumentList();
        allDocs.setDocuments(all);
        tfidf.encode(allDocs);
        Distance distancce = new CosineDistance();
        KMeansImpl kmeans = new KMeansImpl(distancce, k, 10);
        ClusterList clusters = kmeans.cluster(allDocs);
        for(Cluster cluster:clusters.getClusters()){
            DocumentList docs = new DocumentList();
            docs.setDocuments(cluster.getDocuments());
            Keyword keys = new KeyWordImpl((ArrayList<Document>)cluster.getDocuments(), tagger);
            Domain domain = new Domain(docs, keys, cluster.getCentroid());
            this.domainList.add(domain);
        }
        this.save();

    }

    /**
     *
     * @param doc 需要判别领域的文档
     * @param maxDistance 相似度阈值，低于该阈值则认为都不属于已有领域
     * @return 返回所属领域的下标
     */
    @Override
    public int judgeDomain(Document doc, double maxDistance) {
        int index=-1;
        //根据余弦距离判断
        double distance = 1.0;
        WordsVector vec = doc.getTfidf();
//        System.out.println("DomainListImpl.judgeDomain, domainList.size is:"+domainList.size());
        for(int i=0;i<domainList.size();++i){
            WordsVector vec1 = domainList.get(i).getCentroVec();
//            System.out.print(i+"th"+" domain vec is:   ");
//            vec1.printVec();
//            System.out.println("doc vec is:   ");
//            vec.printVec();
//            System.out.println();
            double cosDistance = new CosineDistance().calcDistance(vec, vec1, vec.norm(), vec1.norm());
            if(cosDistance < distance){
                distance = cosDistance;
                index = i;
            }
        }
        if(distance > maxDistance){
            //说明是新领域样本
//            System.out.println("sim is:" + sim);
            index = -1;
        }
        System.out.println("distance:" + distance);
        System.out.println("index:" + index);
        return index;
    }

    @Override
    public void insertDocument(Document doc, double minSim, POSTaggerME tagger) {
        int index = this.judgeDomain(doc, minSim);
        if(index == -1){
            // 说明不属于已有的任何领域,放入缓存文件夹
            if(this.cacheList==null){
                this.cacheList = new DocumentList();
            }
            this.cacheList.updateDocmentList(doc);
            if(cacheList.getDocuments().size() > CACHE_FILE_MAX_NUM){
                // 重新聚类
                int clusterNumNow = this.domainList.size();
                this.updateDomainList(1+ clusterNumNow, tagger);
            }

        }else{
            this.domainList.get(index).updateDomain(doc, tagger);
        }
    }

    @Override
    public void insertDocument(Document doc, int index, POSTaggerME tagger, boolean flag) {
        if(index == -1){
            // 说明不属于已有的任何领域,放入缓存文件夹
            if(this.cacheList==null){
                this.cacheList = new DocumentList();
            }
            this.cacheList.updateDocmentList(doc);
            int num = 0;
            for(Domain domain:this.getDomainList()){
                num += domain.getDocumentList().getDocuments().size();
            }
            num /= this.getDomainList().size();
            if(num < cacheList.getDocuments().size() ){
                // 重新聚类
                System.out.println("超过缓存数量，重新聚类");
                int clusterNumNow = this.domainList.size();
                this.updateDomainList(clusterNumNow + 1, tagger);
            }

        }else{
            this.domainList.get(index).updateDomain(doc, tagger);
        }
    }

    public void save(){
        Constants.writeObjectToFile(this, DOMAINLIST_FILE_PATH);
        System.out.println("保存DomainList模型成功");
    }

    @Override
    public void initial(int k, ArrayList<Document> docs, POSTaggerME tagger) {
        // 第一次创建或者恢复到初始状态
        TfidfEncoder tfidf = new TfidfEncoder();
        DocumentList allDocs = new DocumentList();
        allDocs.setDocuments(docs);
        tfidf.encode(allDocs);
        this.domainList.clear();
        this.cacheList.getDocuments().clear();
        Keyword keys= null;
        DocumentList documents = null;
        if(k == 1){
            keys = new KeyWordImpl((ArrayList<Document>)allDocs.getDocuments(), tagger);
            documents = allDocs;
            Domain domain = new Domain(documents, keys, allDocs.getDocuments().get(0).getTfidf());
            this.domainList.add(domain);
        }else{
            Distance distancce = new CosineDistance();
            KMeansImpl kmeans = new KMeansImpl(distancce, k, 10);
            ClusterList clusters = kmeans.cluster(allDocs);
            for (Cluster cluster : clusters.getClusters()) {
//                System.out.println("a new domain");
                keys = new KeyWordImpl((ArrayList<Document>) cluster.getDocuments(), tagger);
                documents = new DocumentList();
                documents.setDocuments(cluster.getDocuments());
                Domain domain = new Domain(documents, keys, cluster.getCentroid());
                this.domainList.add(domain);
            }
        }
    }

    @Override
    public void printDomainListInfo() {
        int num = this.getDomainList().size();
        System.out.println("********************************打印当前所有领域信息-BEGIN***************************************************");
        System.out.println("领域个数："+num);
        for(int i=0;i<num;++i){
            System.out.println("--------------------------------------------------------------------------------");
            System.out.println("领域"+i +" 信息：");
            System.out.println("文档数量:" + this.getDomainList().get(i).getDocumentList().getDocuments().size());
            System.out.println("文档内容：");
            this.getDomainList().get(i).getDocumentList().printTextInfo();
            System.out.println("关键词信息：");
            this.getDomainList().get(i).getKeyword().printKeywords();
            System.out.println("---------------------------------------------------------------------------------");
        }
        System.out.println("***********************************打印当前所有领域信息-END************************************************");

    }

    public static void main(String[] args) {
        ArrayList<String> strList1 = new ArrayList<String>(){
            {
                add("sb1");
                add("sb2");
            }
        };
        ArrayList<String> strList2 = new ArrayList<>();
        strList2.addAll(strList1);
        strList1.clear();
        System.out.println(strList1.size());
    }

}

