package com.xqq.rep;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

/**
 * Class represents a cluster of documents
 * @author hazoom
 *
 */
public class Cluster implements Serializable{

	private List<Document> documents;
	private WordsVector centroid;
	private double controidNorm;
	
	/**
	 * Constructor initialize a new cluster with one document
	 * @param document
	 */
	public Cluster(Document document) {
		this.centroid = document.getTfidf().clone();
		this.controidNorm = document.getNorm();
		
		this.documents = new ArrayList<Document>();
		this.documents.add(document);
	}
	
	/**
	 * Method add a new document to the cluster
	 * @param document
	 */
	public void add(Document document) {
		this.documents.add(document);
		document.setAssignToCluster(true);
	}
	
	/**
	 * Method updates the centroid of the cluster and updates the norm of the cluster
	 */
	public void updateCentroid() {
		
		this.centroid = null;
		
		for (Document document : this.documents) {
			if (centroid == null) { // First time
				centroid = document.getTfidf().clone();
			} else { // Not first time
				this.centroid.add(document.getTfidf());
			}
		}
		
		this.centroid.divide(this.documents.size());
		this.controidNorm = this.centroid.norm();
	}

	/**
	 *
	 * @param mode:选择输出文本形式(str)或向量形式(vector)
	 */
	public void printDocuments(String mode){

	    List<Document> docs = this.getDocuments();
		for(Document document:docs) {
            if("str".equals(mode)){
                String[] tokens = document.getTokens();  // doc.getTokens()返回一个String[]
                System.out.print("text length:"+tokens.length+", text:");
                for(String token:tokens){
                    System.out.print(token);
                    System.out.print("\t");
                }
            }else{  // 向量形式
                WordsVector vector = document.getTfidf();
                Vector<Double> vec = vector.getData();
                System.out.print("vector length:"+vec.size()+", vector:");
                for(Double d:vec){
                    System.out.print(d);
                    System.out.print("\t");
                }
            }
			System.out.println("");
		}
	}

	public List<Document> getDocuments() {
		return documents;
	}

	public void setDocuments(List<Document> documents) {
		this.documents = documents;
	}

	public WordsVector getCentroid() {
		return centroid;
	}

	public void setCentroid(WordsVector centroid) {
		this.centroid = centroid;
	}

	public double getControidNorm() {
		return controidNorm;
	}

	public void setControidNorm(double controidNorm) {
		this.controidNorm = controidNorm;
	}
}
