package com.xqq;

import java.io.*;

public final class Constants {

    public static final String ROOT_PATH = "D:\\JavaKeywordProject";  // 项目根目录，自行决定
    public static final int CACHE_FILE_MAX_NUM = 5;  // 缓存文件夹文档数量达到多少后进行更新
    public static final int VOCABULARY_SIZE = 3000;  // 词向量维度（TFIDF方法里词汇表长度即词向量维度）
    public static final String EN_TOKEN_PATH = ROOT_PATH + "\\en-token.bin";  // 英文分词模型
    public static final String EN_CHUNKER_PATH = ROOT_PATH + "\\en-chunker.bin";  // 英文句子分块模型（词组）
    public static final String EN_POS_MAXENT_PATH = ROOT_PATH + "\\en-pos-maxent.bin";  // 英文词性标注模型
    public static final String EN_PARSER_CHUNKING = ROOT_PATH + "\\en-parser-chunking.bin";  // 英文依存句法分析
    public static final String TF_IDF_PATH = ROOT_PATH + "\\tfidf_model";  // tfidf 模型
    public static final String CACHE_FILE_PATH = ROOT_PATH + "\\cacheDir";  // 存储缓存文档集合（DocumentList对象）
    public static final String DOMAIN_LIST_PATH = ROOT_PATH + "\\DomainList";  // 存储DomainList对象
    public static final String DEMO_DOCS = ROOT_PATH + "/testDocCluster.txt";

    /**
     *
     * @param obj 保存对象
     * @param fileName 保存路径
     */
    public static void writeObjectToFile(Object obj, String fileName)
    {
        File file =new File(fileName);
        FileOutputStream out;
        try {
            out = new FileOutputStream(file);
            ObjectOutputStream objOut=new ObjectOutputStream(out);
            objOut.writeObject(obj);
            objOut.flush();
            objOut.close();
//            System.out.println("write object success!");
        } catch (IOException e) {
//            System.out.println("write object failed");
            e.printStackTrace();
        }
    }

    /**
     *@param fileName:对象所在文件路径
     * @return 返回一个Object类的对象
     *
     */
    public static Object readObjectFromFile(String fileName)
    {
        Object temp=null;
        File file =new File(fileName);
        FileInputStream in;
        try {
            in = new FileInputStream(file);
            ObjectInputStream objIn=new ObjectInputStream(in);
            temp=objIn.readObject();
            objIn.close();
//            System.out.println("read object success!");
        } catch (IOException e) {
//            System.out.println("read object failed");
//            e.printStackTrace();
        } catch (ClassNotFoundException e) {
//            e.printStackTrace();
        }
        return temp;
    }
}
