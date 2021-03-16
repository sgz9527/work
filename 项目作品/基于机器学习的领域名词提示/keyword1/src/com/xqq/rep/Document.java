package com.xqq.rep;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Class represents a Document
 * @author hazoom
 *
 */
public class Document implements Serializable {
    public static final String SEGMENT_TAG = "cls";
    private String[] tokens;
    private ArrayList<String[]> sentenceList;
    private WordsVector histogram;
    private WordsVector tfidf;
    private double norm;
    private boolean assignToCluster;

    public Document(/*int sentenceId, String topic,*/ String[] tokens, ArrayList<String[]> sentenceList) {
        super();
        this.tokens = tokens;
        this.assignToCluster = false;
        this.sentenceList = sentenceList;
    }

    /**
     *
     * @param tokenList:列表形式的token集合
     */
    public Document(ArrayList<String> tokenList){
        super();
        this.tokens = new String[tokenList.size()];
        int length = tokenList.size();
        for(int i=0;i<length;++i){
            tokens[i] = tokenList.get(i);
        }
        this.sentenceList = new ArrayList<>();
        this.assignToCluster = false;

    }

    public Document(){
        super();
        this.tokens = new String[0];
        this.sentenceList = new ArrayList<>();
        this.assignToCluster = false;
    }

    /**
     * 动态增加该文档的内容
     * @param increaseText:动态加入的文本内容（是新增部分，而不是全部）
     */
    public void updateTokens(String increaseText){
        increaseText = increaseText.replaceAll("[\\pP‘’“”]", "");
        String[] increaseTokens = increaseText.split(" ");
        this.sentenceList.add(increaseTokens);
        int strLen1 = this.tokens.length;// 保存第一个数组长度
        int strLen2 = increaseTokens.length;// 保存第二个数组长度
        this.tokens = Arrays.copyOf(this.tokens, strLen1 + strLen2);// 扩容
        System.arraycopy(increaseTokens, 0, this.tokens, strLen1, strLen2);// 将第二个数组与第一个数组合并
        this.assignToCluster = false;
    }

    public ArrayList<String[]> getSentenceList() {
        return sentenceList;
    }

    public void setSentenceList(ArrayList<String[]> sentenceList) {
        this.sentenceList = sentenceList;
    }
    public String[] getTokens() {
        return tokens;
    }

    public void setTokens(String[] tokens) {
        this.tokens = tokens;
    }

    public WordsVector getHistogram() {
        return histogram;
    }

    public void setHistogram(WordsVector histogram) {
        this.histogram = histogram;
    }

    public double getNorm() {
        return norm;
    }

    public void setNorm(double norm) {
        this.norm = norm;
    }

    public WordsVector getTfidf() {
        return tfidf;
    }

    public void setTfidf(WordsVector tfidf) {
        this.tfidf = tfidf;
    }

    public boolean isAssignToCluster() {
        return assignToCluster;
    }

    public void setAssignToCluster(boolean assignToCluster) {
        this.assignToCluster = assignToCluster;
    }

//    /**
//     * 将tokens以每一行进行存储
//     *
//     * @return
//     */
//    public ArrayList<ArrayList<String>> segmentInToSentenceList() {
//        ArrayList<ArrayList<String>> sentenctList = new ArrayList<>();
//        ArrayList<String> strs = new ArrayList<>();
//        for (int i = 0; i < this.tokens.length; ++i) {
//            if (SEGMENT_TAG.equals(this.tokens[i])) {
//                sentenctList.add(strs);
//                strs.clear();
//            } else {
//                strs.add(this.tokens[i]);
//            }
//        }
//        sentenctList.add(strs);
//        return sentenctList;
//    }
public static void main(String[] args) {
    String sb = "";
    String[] res = sb.split(" ");
    System.out.println(res.length);
}
}
