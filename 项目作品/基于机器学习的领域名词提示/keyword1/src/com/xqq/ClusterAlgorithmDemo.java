package com.xqq;

import com.xqq.api.Encoder;
import com.xqq.impl.CosineDistance;
import com.xqq.impl.KMeansImpl;
import com.xqq.impl.TfidfEncoder;
import com.xqq.rep.Cluster;
import com.xqq.rep.ClusterList;
import com.xqq.rep.Distance;
import com.xqq.rep.DocumentList;

import java.util.ArrayList;


/**
 * ClusterAlgorithmDemo class to run k-means algorithm(demo)
 * @author hazoom
 *
 */
public class ClusterAlgorithmDemo {

	public static void main(String[] args) {
	    String fileName = Constants.DEMO_DOCS;

		DocumentList documents = new DocumentList(fileName);  // 进行数据清洗

		System.out.println("Finish preprocessing...");
		
		Encoder encoder = new TfidfEncoder(false);  // 转为数字向量
		encoder.encode(documents);

		System.out.println("Finish encoding...");
		
		Distance distancce = new CosineDistance();
		
		KMeansImpl kmeans = new KMeansImpl(distancce, 2, 10);
		ClusterList clusters = kmeans.cluster(documents);
		
		System.out.println("Finish K-means algorithm...");
		
		int i = 1;
		for (Cluster cluster : clusters.getClusters()) {
			System.out.println("Cluster no. " + i + " has " + cluster.getDocuments().size() + " documents.");
			cluster.printDocuments("str");
			i++;
		}
	}

}
