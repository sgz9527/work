package com.xqq;

import com.xqq.api.DomainList;
import com.xqq.api.Keyword;
import com.xqq.impl.Domain;
import com.xqq.impl.DomainListImpl;
import com.xqq.impl.TfidfEncoder;
import com.xqq.rep.Document;
import com.xqq.rep.DocumentList;
import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.parser.Cons;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * MainAction ：整个事件过程，包括初始化，领域判别，领域更新，关键词提取， 关键词搜索。
 * @author xqq
 */
public class MainAction {
    public static final Double MAX_DISTANCE = 0.6;
    public static final String TAG_MODEL_PATH = Constants.EN_POS_MAXENT_PATH;
    private DomainList domainList;
    private int belongDomainIndex;
    private POSTaggerME tagger;
    private TfidfEncoder tfidf;
    private Document alreadyInput;  // 用于保存当前用户所有已经输入的部分

    public Document getAlreadyInput() {
        return alreadyInput;
    }

    public void setAlreadyInput(Document alreadyInput) {
        this.alreadyInput = alreadyInput;
    }

    public TfidfEncoder getTfidf() {
        return tfidf;
    }

    public void setTfidf(TfidfEncoder tfidf) {
        this.tfidf = tfidf;
    }

    public int getBelongDomainIndex() {
        return belongDomainIndex;
    }

    public void setBelongDomainIndex(int belongDomainIndex) {
        this.belongDomainIndex = belongDomainIndex;
    }

    public DomainList getDomainList() {
        return domainList;
    }

    public void setDomainList(DomainList domainList) {
        this.domainList = domainList;
    }
    public MainAction(){
        this.domainList = (DomainListImpl)Constants.readObjectFromFile(Constants.DOMAIN_LIST_PATH);
        this.belongDomainIndex = -1;
        this.tfidf = new TfidfEncoder();
        this.alreadyInput = new Document();
        File file = new File(TAG_MODEL_PATH);
        POSModel model = new POSModelLoader().load(file);
        this.tagger = new POSTaggerME(model);
    }

    /**
     * 判断文档所属领域
     * 用户编辑文档时调用（当用户编辑完基本部分后不再调用该函数，在这之前需要动态判断，可以每输入一个句子判断一次）
     * 注意，updateTokens(increaseInput)需要一直执行，可以每一个句子更新一次，直到退出
     */
    public void judgeDomain(){
        if(this.domainList != null) {
            this.tfidf.encode(alreadyInput);
            this.belongDomainIndex = this.domainList.judgeDomain(alreadyInput, MAX_DISTANCE);
        }
    }


    /**
     * 更新用户输入
     * @param increaseInput:新增的输入
     */
    public void updateTokens(String increaseInput){
        this.alreadyInput.updateTokens(increaseInput);
    }

    /**
     * 根据已输入内容，以及当前正在输入的前缀，进行关键词推荐。根据监听事件触发
     *  例如："there is a veh"，其中 "there is a"为preContext，"veh"为prex(vehicle的前缀)
     * @param preContext: 同一个句子中，除了正在输入的部分(prex）
     * @param prex:正在输入的部分
     */
    public ArrayList<String> searchKeyword(String preContext, String prex){
        Domain domain;
        try {
            domain = ((DomainListImpl) this.domainList).getDomainList().get(this.belongDomainIndex);
        }catch(NullPointerException e){
            System.out.println("不存在 domainList");
            return new ArrayList<>();
        }catch(ArrayIndexOutOfBoundsException e){
            System.out.println("belongDomainIndex = -1, forbid searchKeyword function");
            return new ArrayList<>();
        }
        Keyword keyword = domain.getKeyword();
        return keyword.keywordSearch(preContext, prex, tagger);
    }

    /**
     * 领域更新（插入文档或重新聚类）
     * 在用户退出编辑界面并保存后执行
     */
    public void insertDomain(){
        if(this.tfidf.getIdf()==null){
            List<Document> docs = new ArrayList<>();
            DocumentList documentList = new DocumentList();
            docs.add(this.alreadyInput);
            documentList.setDocuments(docs);
            this.tfidf.encode(documentList);
        }
        this.tfidf.encode(this.alreadyInput);
        try{
            this.domainList.insertDocument(alreadyInput,belongDomainIndex, this.tagger, true);
        }catch(NullPointerException e){
            System.out.println("不存在 domainList, 现在创建");
            ArrayList<Document> docs = new ArrayList<>();
            docs.add(this.alreadyInput);
            this.domainList = new DomainListImpl(1, docs, tagger);
        }
    }
    private void testRun(String[] sentences){
        for(int i=0;i<6;++i){
            System.out.println();
        }
        System.out.println("******************************************************testRun-BEGIN*************************************************************************");
        //模拟运行
        for(String s:sentences){
            this.updateTokens(s);
        }
        String[] tokens = this.alreadyInput.getTokens();
        System.out.println("tokens is: ");
        for(String token:tokens){
            System.out.print(token + " ");
        }
        System.out.println();
        this.judgeDomain();
        System.out.println("belong domain index is："+this.belongDomainIndex);
        String preContext = "for vehicle to";
        String prex = "d";
        ArrayList<String> res = this.searchKeyword(preContext, prex);
        System.out.println("结果为:");
        for(String keyword:res){
            System.out.println(keyword);
        }

        this.insertDomain();
        this.save();
        DomainListImpl domainList = (DomainListImpl)(this.domainList);
        System.out.println();
        System.out.println();
        domainList.printDomainListInfo();
        System.out.println();
        System.out.println();
        System.out.println("cache file num is: "+domainList.getCacheList().getDocuments().size());
        System.out.println("******************************************************testRun-END***************************************************************************");

    }
    private void test(){
        List<Integer> list = new ArrayList<>();
        for(int i=0;i<1000000;++i){
            list.add(i);
        }
        for(int i=1000000-1;i>=0;--i) {
            list.remove(list.size()-1);
        }
    }

    public void save(){
        this.domainList.save();
    }
    public static void main(String[] args) {
        String fileName = Constants.DEMO_DOCS;
        DocumentList documents = new DocumentList(fileName);
        List<Document> documentList = documents.getDocuments();
        MainAction a = new MainAction();
        for(Document doc:documentList){
            List<String[]> sentencesList = doc.getSentenceList();
            List<String> testSentences = new ArrayList<>();//获取测试模式下的输入句子列表
            for(String[] sens:sentencesList){
                StringBuilder sb = new StringBuilder("");
                for(String s:sens){
                    sb.append(s + " ");
                }
                sb.delete(sb.length()-1, sb.length());
                testSentences.add(sb.toString());
            }
//            System.out.println("---------------------------------doc-----------------------------------------");
            String[] ss = new String[testSentences.size()];
            for(int i=0;i<ss.length;++i) {
                ss[i] = testSentences.get(i);
//                System.out.println(ss[i]);
            }
            a.testRun(ss);
        }

    }
}
