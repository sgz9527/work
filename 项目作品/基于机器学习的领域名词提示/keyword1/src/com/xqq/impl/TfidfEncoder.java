package com.xqq.impl;

import com.xqq.Constants;
import com.xqq.api.Encoder;
import com.xqq.rep.Document;
import com.xqq.rep.DocumentList;
import com.xqq.rep.WordsVector;

import java.io.Serializable;


/**
 * TF-IDF encoder
 * @author hazoom,xqq
 *
 */
public class TfidfEncoder implements Encoder, Serializable{

    public static String getTfIdfPath() {
        return TF_IDF_PATH;
    }

    public static int getVocabularySize() {
        return VOCABULARY_SIZE;
    }

    public boolean isFirst() {
        return isFirst;
    }

    public WordsVector getIdf() {
        return idf;
    }

    public void setIdf(WordsVector idf) {
        this.idf = idf;
    }

    private static final String TF_IDF_PATH = Constants.TF_IDF_PATH;
	private static final int VOCABULARY_SIZE = Constants.VOCABULARY_SIZE;
	private boolean isFirst=true;
	private WordsVector idf = null;
	
	public TfidfEncoder() {
	    //初始化函数判断是否第一次运行，如果是第一次运行则计算idf，否则直接从文件加载
        this.loadModel();
//        System.out.println(isFirst);
    }
	public TfidfEncoder(boolean load){
		if(load){
			loadModel();
		}else{
		    this.isFirst = true;
        }
	}
	@Override
	public void encode(DocumentList documents) {
		
		this.calcHistogram(documents);
//        System.out.println(isFirst);
        if(isFirst|| this.idf==null){
            System.out.println("第一次加载,计算idf");
            this.calcIDF(documents);
            this.saveModel();
            isFirst = false;
        }
		for (Document document : documents.getDocuments()) {
			this.encode(document);
		}
	}

	/**
	 * 保存编码模型参数，下次直接加载
	 */
	@Override
	public void saveModel() {
        // 保存this.idf
        Constants.writeObjectToFile(this.idf, TfidfEncoder.TF_IDF_PATH);
	}

	/**
	 * 从文件加载保存的编码模型参数，前提是存在（不存在则说明是第一次运行）
	 * @return
	 */
	@Override
	public void loadModel() {
        this.idf = (WordsVector) Constants.readObjectFromFile(TfidfEncoder.TF_IDF_PATH);
        this.isFirst = this.idf == null;
	}

	@Override
	public void updateModel() {

	}

	/**
	 * Method calculates the words histogram for the given document
	 * @param document
	 */
	private void calcHistogram(Document document) {
		
		WordsVector hist = new WordsVector(VOCABULARY_SIZE);
		
		String[] tokens = document.getTokens();
		for (String token : tokens) {
			int hash = Math.abs(token.hashCode()) % VOCABULARY_SIZE;
			hist.increment(hash);
		}
		
		document.setHistogram(hist);
	}
	
	/**
	 * Method calculates words histogram for a given document list
	 * @param documents: 文档对象列表
	 */
	private void calcHistogram(DocumentList documents) {
		for (Document document : documents.getDocuments()) {
			this.calcHistogram(document);
		}
	}
	
	/**
	 * Method calculates the inverse document frequency term of the documents in the corpus
	 * @param documents
	 */
	private void calcIDF(DocumentList documents) {
		
		this.idf = new WordsVector(VOCABULARY_SIZE);
//        int count = 0;
        if(documents.getDocuments().size() == 1){
            for(int i=0;i<VOCABULARY_SIZE;++i){
                this.idf.increment(i);
            }
            return;
        }
		for (Document document : documents.getDocuments()) {
//            System.out.println("doc:");
            for (int i = 0; i < VOCABULARY_SIZE; i++) {
                if (document.getHistogram().get(i) > 0) {
//                    System.out.println(document.getHistogram().get(i));
					//Increment the count of seen that word in whole documents
					this.idf.increment(i);
//					++count;
				}
			}
		}
//		this.idf.printVec();
//        System.out.println("total word num:"+count);

        // Multiply by the number of documents
		this.idf.multiply(documents.getDocuments().size());
//        System.out.println("after multiply:");
//        this.idf.printVec();
		// Log the array
        this.idf.log();
//        System.out.println("after log, idf is: ");
//        this.idf.printVec();

	}
	
	/**
	 * Method encodes the tf-idf score for a given document
	 * @param document:Document 对象
	 */
	public void encode(Document document) {
		WordsVector tfidf = null;
		if(document.getHistogram()!=null){
			tfidf = document.getHistogram().clone();
		}else{
			this.calcHistogram(document);
            tfidf = document.getHistogram().clone();
		}
		
		// No need of the histogram no more
//		document.setHistogram(null);
		
		// Divide by vector's max (normalization)
		tfidf.divide(tfidf.max());
		
		// Dot product of tf and idf vectors
        try {
            tfidf.multiply(this.idf);  // 需要保存的是逆文档概率部分
        }catch(NullPointerException e){
            System.out.println("第一次运行tf-idf，未训练");

//            e.printStackTrace();
            return;
        }
		// Set the tf-idf score and norm of this document
		document.setTfidf(tfidf);
		document.setNorm(tfidf.norm());
	}

}
