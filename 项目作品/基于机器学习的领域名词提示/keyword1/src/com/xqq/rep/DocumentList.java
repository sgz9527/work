package com.xqq.rep;

import com.xqq.Constants;
import com.xqq.api.Keyword;
import com.xqq.impl.KeyWordImpl;
import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;

import javax.print.Doc;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class DocumentList implements Serializable{

	private List<Document> documents;

	public DocumentList(){
	    documents = new ArrayList<>();
    }
	
	/**
	 * Constructor reads input file and creates the list of documents
	 * @param filename
	 */
	public DocumentList(String filename) {
		
		this.documents = new ArrayList<>();
		
		BufferedReader br = null;
		
		try {
			br = new BufferedReader(new FileReader(filename));
			String line = null;
			while ((line = br.readLine()) != null) {
				if (!line.isEmpty()) {
				    String text;
                    String text1 = line.replaceAll("[\\pP‘’“”]", "");
                    text = line.replaceAll("[,.]", Document.SEGMENT_TAG);
//                    System.out.println("tag the sentence:" + text);
                    String[] tokens;
                    tokens = text1.split(" " );  // 按单词分开的tokens,不含句子分割符
                    ArrayList<String[]> sentenceList= new ArrayList<>();  //按句子分开
                    String[] sentences = text.split(Document.SEGMENT_TAG);
                    for(String sentence:sentences){
                        sentence = sentence.replaceAll("[\\pP‘’“”]", "");
                        String[] words = sentence.split(" ");
                        sentenceList.add(words);
                    }
                    Document doc = new Document(tokens, sentenceList);
                    this.documents.add(doc);
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

    public DocumentList(String[] docs){  // 直接从字符串数组中
	    this.documents = new ArrayList<Document>();
	    for (String text:docs){
            String text1 = text.replaceAll("[\\pP‘’“”]", "");
            text = text.replaceAll("[,.]", "");
//            System.out.println("tag the sentence:" + text);
            String[] tokens;
            tokens = text1.split(" " );  // 按单词分开的tokens,不含句子分割符
            ArrayList<String[]> sentenceList= new ArrayList<>();  //按句子分开
            String[] sentences = text.split(Document.SEGMENT_TAG);
            for(String sentence:sentences){
                sentence = sentence.replaceAll("[\\pP‘’“”]", "");
                String[] words = sentence.split(" ");
                sentenceList.add(words);
            }
            Document doc = new Document(tokens, sentenceList);
	        this.documents.add(doc);
        }
    }
    public DocumentList(ArrayList<String> docs){  // 直接从字符串List中
        this.documents = new ArrayList<Document>();
        for (String text:docs){
            String text1 = text.replaceAll("[\\pP‘’“”]", "");
            text = text.replaceAll("[,.]", Document.SEGMENT_TAG);
//            System.out.println("tag the sentence:" + text);
            String[] tokens;
            tokens = text1.split(" " );  // 按单词分开的tokens,不含句子分割符
            ArrayList<String[]> sentenceList= new ArrayList<>();  //按句子分开
            String[] sentences = text.split(Document.SEGMENT_TAG); // 将输入文本按逗号和句号分割成一个一个的句子
            if(sentences.length == 0){
                sentences = new String[1];
                sentences[0] = text;
            }
            for(String sentence:sentences){
                if(!sentence.isEmpty()){
                    if(" ".equals(sentence.substring(0,1))){
                        sentence = sentence.substring(1);
                    }
                }
                sentence = sentence.replaceAll("[\\pP‘’“”]", "");
                String[] words = sentence.split(" ");
                sentenceList.add(words);
            }
            System.out.println("tokens.length:"+tokens.length);
            Document doc = new Document(tokens, sentenceList);
            this.documents.add(doc);
        }
    }
	public List<Document> getDocuments() {
		return documents;
	}

	public void setDocuments(List<Document> documents) {
		this.documents = documents;
	}

	public void updateDocmentList(Document doc){
        this.documents.add(doc);
    }

	public void printDocumentTexts(List<Document> docs){
        for(Document doc:docs) {
            String[] tokens = doc.getTokens();  // doc.getTokens()返回一个String[]
            for(String token:tokens){
                System.out.print(token + " ");
            }
            System.out.println();

        }
    }
    public void printTextInfo(){
	    ArrayList<Document> docs = (ArrayList<Document>)this.getDocuments();
        for(Document doc:docs) {
            String[] tokens = doc.getTokens();  // doc.getTokens()返回一个String[]
            for(String token:tokens){
                System.out.print(token + " ");
            }
            System.out.println("");

        }
    }
    public void printDocumentTextsSentenceList(List<Document> docs){
	    for(Document doc:docs){
	        ArrayList<String[]> sentenceList = doc.getSentenceList();
            System.out.println("----------------------------------------------------------------------------");
            for(String[] words: sentenceList){
                System.out.println("");
                for(String word:words){
                    System.out.print(word + " ");
                }
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        String fileName = Constants.DEMO_DOCS;
        ArrayList<String> texts = new ArrayList<String>(){
            {
                add("vehicle car drive,you should know these skills.");
                add("train car drive, is a ? hello > turn left.!");
                add("turn left car drive");
                add("natural language processing");
            }
        };
        DocumentList docsObj = new DocumentList(fileName);
        List<Document> docs = docsObj.getDocuments();
//        docsObj.printDocumentTexts(docs);
//        docsObj.printDocumentTextsSentenceList(docs);
        File file = new File(Constants.EN_POS_MAXENT_PATH);
        POSModel model = new POSModelLoader().load(file);
        POSTaggerME tagger = new POSTaggerME(model);
        Keyword keyword = new KeyWordImpl((ArrayList<Document>)docs, tagger);
//        keyword.keywordExtraction(docs.get(2));
        keyword.testKeywordSearch(tagger);

    }
}
