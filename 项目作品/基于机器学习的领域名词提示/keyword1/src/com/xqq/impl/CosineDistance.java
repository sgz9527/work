package com.xqq.impl;

import com.xqq.rep.Distance;
import com.xqq.rep.WordsVector;

import java.io.Serializable;

/**
 * Class calculates the cosine distance between two vectors
 * @author hazoom
 *
 */
public class CosineDistance extends Distance implements Serializable{

	@Override
	public double calcDistance(WordsVector vector1, WordsVector wv2, double norm1, double norm2) {
		
		return 1.0 - (vector1.dotProduct(wv2) / (norm1*norm2));
	}

}
