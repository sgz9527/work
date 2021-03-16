package com.xqq.impl;


import com.xqq.Constants;
import com.xqq.api.Keyword;
import com.xqq.rep.Document;
import com.xqq.rep.WordsVector;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.Span;

import java.io.*;
import java.util.*;

/**
 * 此类提供关键词相关功能，包括关键词提取，关键词搜索
 *
 */
public class KeyWordImpl implements Keyword ,Serializable{
    private static final String HASHMAP_KEYWORD = "keyWord";
    private static final String HASHMAP_TYPE = "type";
    private static final String TAG_NP = "NP";
    private static final String TAG_VP = "VP";
    private static final String VP_SEQUENCE_PATH = Constants.ROOT_PATH + "/vpSequence.txt";
    private HashMap<String, ArrayList<String> >  keywords;  // 得到的最终关键词集合,按不同词性存储
    private int keywordsNum;  // 提取出来的关键词信息包含的单词数量
    private HashSet<String> keyWordPhraseSet;


    /*

    例如：keywords = {
                "np":["", "",...],
                "vp":["", "",...]
        }

     */





    public HashMap<String, ArrayList<String>> getKeywords() {
        return keywords;
    }

    public void setKeywords(HashMap<String, ArrayList<String>> keywords) {
        this.keywords = keywords;
    }
    public int getKeywordsNum() {
        return keywordsNum;
    }

    public void setKeywordsNum(int keywordsNum) {
        this.keywordsNum = keywordsNum;
    }
    public HashSet<String> getKeyWordPhraseSet() {
        return keyWordPhraseSet;
    }

    public void setKeyWordPhraseSet(HashSet<String> keyWordPhraseSet) {
        this.keyWordPhraseSet = keyWordPhraseSet;
    }

    public KeyWordImpl(ArrayList<Document> docs, POSTaggerME tagger) {
        //构造关键词表，索引表，中心向量
        this.keywords = new HashMap<>();
        this.keywordsNum = 0;
        this.keyWordPhraseSet = new HashSet<>();
        this.keywordExtraction(docs, tagger);
    }


    /**
     *
     */
    @Override
    public void keywordExtraction(Document doc, POSTaggerME tagger) {
        /*
        1.获取文本 text
        2.对text分词，得到tokens
        3.利用opennlp分块模型得到结果res1
        4.利用规则进行提取，得到结果re2
        5.过滤，去重
         */
        TfidfEncoder tfidf = new TfidfEncoder();
        tfidf.encode(doc);
        int count = 0;
        Double meanTfidfVal = 0.0;  // 文档中出现的单词的平均tfidf值，低于该值的过滤掉
        Vector<Double> tf = doc.getHistogram().getData();
        Vector<Double> tfidfVal = doc.getTfidf().getData();
        for(int i=0;i<Constants.VOCABULARY_SIZE;++i){
            if(tf.get(i) != 0.0){
//                System.out.println(tfidfVal.get(i));
                meanTfidfVal += tfidfVal.get(i);
                ++count;
            }
        }
        meanTfidfVal /= count*2;
        ArrayList<String[]> sentences = doc.getSentenceList();
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(Constants.EN_CHUNKER_PATH);
        }catch(IOException e){
            e.printStackTrace();
        }
        ChunkerModel chunkerModel=null;
        try {
            chunkerModel = new ChunkerModel(inputStream);
        }catch(IOException e){
            e.printStackTrace();
        }
        ChunkerME chunkerME = new ChunkerME(chunkerModel);
        ArrayList<String[]> tagsList= new ArrayList<>(); // 每一个句子的标签组成的集合
        ArrayList<HashMap<String, Object>> keywordsListOpenNlp = new ArrayList<>(); // 每一个句子中的关键词词组组成的集合
        ArrayList<HashMap<String, Object>> keywordsListRule = new ArrayList<>();  // 根据规则进行提取动词短语

        for(String[] sentence:sentences){
            String[] tags = tagger.tag(sentence);
            tagsList.add(tags);
            Span[] span = chunkerME.chunkAsSpans(sentence, tags);
            Span[] spanByRule = chunkInfoByRule(tags);
            generateKeywordsList(span,tags,sentence,meanTfidfVal,doc,keywordsListOpenNlp);  // openNlp提取出来的包括名词短语和动词短语
            generateKeywordsList(spanByRule, tags, sentence,meanTfidfVal, doc, keywordsListRule);  // 通过规则提取出来的全是动词短语
//            System.out.println(keywordsListRule.size());
//            System.out.println();
            //合并，去重
            for(int i=0;i<keywordsListOpenNlp.size();++i){
                // 将opennlp里的动词短语移到规则提取的集合里，则：openNlp集合里全是名词短语，规则提取的集合里全是动词短语
                if(TAG_VP.equals(keywordsListOpenNlp.get(i).get(HASHMAP_TYPE))){
                    keywordsListRule.add((HashMap<String, Object>)keywordsListOpenNlp.get(i).clone());
                    keywordsListOpenNlp.remove(i--);
                }

            }
//            System.out.println("去重前******************************************************************************");
//            this.printKeywordsList(keywordsListOpenNlp);
//            this.printKeywordsList(keywordsListRule);
            this.eraser(keywordsListOpenNlp);
            this.eraser(keywordsListRule);
//            System.out.println("去重后:****************************************************************************");
//            this.printKeywordsList(keywordsListOpenNlp);
//            this.printKeywordsList(keywordsListRule);
            // 生成最终结果
            int nums1 = AddIntoFinalKeywordList(keywordsListOpenNlp);
            int nums2 = AddIntoFinalKeywordList(keywordsListRule);
            int increaseNum = nums1 + nums2;  //新增关键词组个数
            this.keywordsNum += increaseNum;

//            printKeywords();

        }

//        System.out.println("这份文档总共 "+keywordsListChunkerModel.size()+" 个关键词组");


    }
    @Override
    public void keywordExtraction(ArrayList<Document> docs, POSTaggerME tagger){
        for(Document doc:docs){
            this.keywordExtraction(doc, tagger);
        }
    }

    private ArrayList<String> obtainKeywordTokens(ArrayList<HashMap<String, Object>> keywordsList) {
        ArrayList<String> keywords = new ArrayList<>();
        for(HashMap<String, Object> keywordPhrase:keywordsList){
            ArrayList<String[]> keywordInfo = (ArrayList<String[]>)keywordPhrase.get(HASHMAP_KEYWORD);
            for(String[] keyword:keywordInfo){
                keywords.add(keyword[1]); // keyword[0]为词性， 1为文本
            }
        }
        return keywords;
    }

    private void printKeywordsList(ArrayList<HashMap<String, Object>> keywordsList){
        for(HashMap<String, Object> keywordPhrase:keywordsList){
            this.printOneKeywordPhrase(keywordPhrase);
        }
        System.out.println();
    }
    public void printKeywords(){
        System.out.println("print keywords info.The first row is the tag of keywords, second row is the keyword phrase which is divided by '|'.");
        Set<String> keys = this.keywords.keySet();
        for(String key:keys){
            System.out.println(key);
            ArrayList<String> keywords = this.keywords.get(key);
            this.printKeywords(keywords);
            System.out.println();
            System.out.println("");

        }
    }
    public void printKeywords(ArrayList<String> keywords){
        for(String keyword:keywords){
            System.out.print(keyword + "   |    ");
        }
    }

    private int AddIntoFinalKeywordList(ArrayList<HashMap<String, Object>> keywordsList){
        int count = 0;
        for(int i=0;i<keywordsList.size();++i){
            ArrayList<String[]> keywords = (ArrayList<String[]>) keywordsList.get(i).get(HASHMAP_KEYWORD);
            String keyword = this.concatAllWords(keywords, " ");
            if(this.keyWordPhraseSet.contains(keyword)){
                keywordsList.remove(i--);
                continue;
            }else{
                this.keyWordPhraseSet.add(keyword);
            }
            String key = keywordsList.get(i).get(HASHMAP_TYPE) + "-" + keywords.get(0)[0];
            if(this.keywords.containsKey(key)){
                this.keywords.get(key).add(keyword);
            }else{
                ArrayList<String> temp = new ArrayList<>();
                temp.add(keyword);
                this.keywords.put(key, temp);
            }
            count++;
        }
        return count;
    }
    private void eraser(ArrayList<HashMap<String, Object>> keywordsList){
        HashSet<String> keywordSet = new HashSet<>();
        for(int i=0;i<keywordsList.size();++i){
            String s = this.concatAllWords((ArrayList<String[]>)keywordsList.get(i).get(HASHMAP_KEYWORD));
            if(keywordSet.contains(s)){
                keywordsList.remove(i--);
            }else{
                keywordSet.add(s);
            }
        }
    }

    private String concatAllWords(ArrayList<String[]> keywordInfo){
        StringBuilder sb = new StringBuilder();
        for(String[] keyword:keywordInfo){
            sb.append(keyword[1]);
        }
        return sb.toString();
    }
    private String concatAllWords(ArrayList<String[]> keywordInfo, String join){
        StringBuilder sb = new StringBuilder();
        for(String[] keyword:keywordInfo){
            sb.append(keyword[1]);
            sb.append(join);
        }
        sb.delete(sb.length()-join.length(), sb.length());
        return sb.toString();
    }
    private void generateKeywordsList(Span[] span, String[] tags, String[] sentence, Double meanTfidfVal,
                                      Document doc, ArrayList<HashMap<String, Object>> keywordsList){
        for(Span s:span){  // 一个span代表一个关键词词组
            HashMap<String, Object> temp = new HashMap<>();  // 存储一个关键词词组，包括词组属性以及对应的关键字集合
//                System.out.print(s.getStart() + " "+ s.getEnd() + " " + s.getType()+ "  |  ");
            if(!TAG_NP.equals(s.getType())&& !TAG_VP.equals(s.getType())){
                //过滤非动词短语和名词短语的其他无用成分
                continue;
            }
            temp.put(HASHMAP_TYPE,s.getType());
            ArrayList<String[]> keywordInfo = new ArrayList<>();  // 1个关键词词组(关键词集合以及对应的属性）
            for(int j = s.getStart(); j< s.getEnd(); ++j){
                String[] key = new String[2];
                key[0] = tags[j];
                key[1] = sentence[j];
                keywordInfo.add(key);
            }
            temp.put(HASHMAP_KEYWORD, keywordInfo);
            this.filterKeywordPhrase(temp);
            if(((ArrayList<String[]>)temp.get(HASHMAP_KEYWORD)).size()==0){
                continue;
            }
            Double tfidfValue = this.calTfidfOfKeywords(temp, doc);  // 该关键词词组当中最大的tfidf值
            if(tfidfValue < meanTfidfVal){
                //过滤tfidf值太小的词组
                continue;
            }
//            this.printOneKeywordPhrase(temp);
            keywordsList.add(temp);
        }
    }
    private Span[] chunkInfoByRule(String[] tags){
        ArrayList<String[]> vpTagList = new ArrayList<>();
        BufferedReader br = null;
        try{
            br = new BufferedReader(new FileReader(VP_SEQUENCE_PATH));
        }catch(FileNotFoundException e){
            e.printStackTrace();
        }
        String line = null;
        try {
            while ((line = br.readLine()) != null) {
                if (!line.isEmpty()) {
                    String[] vpTags = line.split(" ");
                    vpTagList.add(vpTags);
                }
            }
        }catch(IOException e){
            e.printStackTrace();
        }
        ArrayList<Span> spanList = new ArrayList<>();
        for(String[] vpTags:vpTagList){
            searchPattern(vpTags, tags, spanList);
        }
        Span[] spans = new Span[spanList.size()];
        int i = 0;
        for(Span s:spanList){
            spans[i++] = s;
        }
        return spans;

    }
    private static void searchPattern(String[] vpTags, String[] tags, ArrayList<Span> spanList){
        if(vpTags.length <= tags.length) {
            for (int i = 0; i < tags.length; ++i) {

                if (vpTags[0].equals(tags[i])) {
//                    System.out.println(tags[i]+"=="+vpTags[0]);
                    int k = i + 1;
                    int j=1;
                    while (j < vpTags.length && k < tags.length && vpTags[j].equals(tags[k])){
//                        System.out.println(tags[k]+"=="+vpTags[j]);
                        ++j;
                        ++k;
                    }
                    if (j < vpTags.length) {
                        continue;
                    }
                    Span s = new Span(i, k, TAG_VP);
                    spanList.add(s);
                }
            }
        }
    }

    private Double calTfidfOfKeywords(HashMap<String, Object> temp, Document doc) {
        ArrayList<String[]> keywordInfo = (ArrayList<String[]>) temp.get(HASHMAP_KEYWORD);
        Double maxVal = 0.0;
        for(String[] keyword:keywordInfo){
            int hash = Math.abs(keyword[1].hashCode()) % TfidfEncoder.getVocabularySize();  // 该关键词对应于tfidf向量中的索引
            Double tfidfVal = doc.getTfidf().getData().get(hash);
            if(maxVal<tfidfVal){
                maxVal = tfidfVal;
            }
        }
        return maxVal;
    }

    public void printOneKeywordPhrase(HashMap<String, Object> temp){
        String type = (String)temp.get(HASHMAP_TYPE);
        System.out.print(type+"{  ");
        ArrayList<String[]> keywordInfo = (ArrayList<String[]>) temp.get(HASHMAP_KEYWORD);
        for(String[] keyword:keywordInfo){
            System.out.print("("+ keyword[0] + ")" + keyword[1]+" ");
        }
        System.out.print("  }  |  ");
    }

    /**
     * 对于一个句子中的所有关键词组进行修正
     * @param temp：一个句子中的关键词信息，包含关键词词组属性（NP或VP）以及对应的单词序列（以及对应的词性标签）
     */
    private void filterKeywordPhrase(HashMap<String, Object> temp){
        ArrayList<String[]> keywordInfo = (ArrayList<String[]>) temp.get(HASHMAP_KEYWORD);  // 1个关键词词组
//        例如 [["DT","the"],["NN","vehicle"]]
        if(keywordInfo.size()>0){
            String[] firstWord = keywordInfo.get(0);
            if(!firstWord[0].isEmpty()){
                if(firstWord[0].matches("[WP]*DT")){
                    keywordInfo.remove(0);
                }else if(firstWord[0].matches("JJ")){
//                    System.out.print("形容词+名词");
                }else if(!firstWord[1].isEmpty()){
                    if(firstWord[1].matches("am|is|are|were|was|be|been")){
                        keywordInfo.remove(0);
                    }
                }
            }
        }

    }

    /**
     *
     * @param predictTags:当前输入可能的词性
     * @param prex 需要搜索的单词前缀
     */
    @Override
    public ArrayList<String> keywordSearch(String[] predictTags, String prex){
        ArrayList<String> result = new ArrayList<>();
        if(predictTags==null){
            return result;
        }
        for(String tag:predictTags){
            System.out.println("predict tags:" + tag);
            if("NP".equals(tag)){  // 如果是搜索所有的名词
                result.addAll(searchAll(prex, "(.*)NP(.*)"));
            }else if("VP".equals(tag)){
                result.addAll(searchAll(prex, "(.*)VP(.*)"));
            }
            else if("TNP".equals(tag)){
                ArrayList<String> temp = searchAll(prex,"(.*)NP(.*)");
                for(int i=0;i<temp.size();++i){
                    temp.set(i, "The" + temp.get(i));
                }
                result.addAll(temp);
            }else {
                result.addAll(search(prex, this.keywords.get(tag)));
            }
        }
        return result;
    }

    @Override
    public ArrayList<String> keywordSearch(String preContext, String prex, POSTaggerME tagger) {
        preContext = preContext.replaceAll("[\\pP‘’“”]", "");
        String[] preTokens = preContext.split(" ");
        String[] preTags = tagger.tag(preTokens);
        String[] predictTags = this.predictWordTag(preTokens, preTags);
        return this.keywordSearch(predictTags, prex);
    }

    @Override
    public void updateKeyword(Document doc, POSTaggerME tagger) {
        this.keywordExtraction(doc, tagger);

    }

    public String[] predictWordTag(String[] preTokens, String[] preTags){
        /*
        定义判断规则，根据前几个单词预测接下来的单词词性（可通过训练二分类模型进行优化）
         */
        String[] tags = null;  // 所有可能的词性标签，总共两层，第一层为大类，第二层为细类
        int length = preTokens.length;
        String preWord = preTokens[length-1];
        String preTag = preTags[length-1];
        System.out.println("preTag:" + preTag);
        if(preWord.matches("am|is|are|were")){
            tags = new String[]{
                    "VP-VBG","NP-JJ","NP-DT"
            };  // 例如is doing, is perfect, is a
        } else if(length >= 3&&preTokens[length-1].equals("CASE")&&preTokens[length-2].equals("USE")&&preTokens[length-3].equals("INCLUDE")){
            tags = new String[]{
                    "NP"
            };
        }else if(length == 0){  // 前面没有单词,提示为"The"+名词短语
            tags = new String[]{
                    "TNP"
            };
        }
        else if(preTag.matches("[P|W]*DT")){
            tags = new String[]{
                    "NP-NN","NP-JJ","NP-DT"
            };
        }else if(preTag.matches("JJ")){
            tags = new String[]{
                    "NP-NN"
            };
        }else if(preTag.equals("TO")){
            tags = new String[]{
                    "VP"
            };
        }
        return tags;
    }

    private ArrayList<String> search(String prex, ArrayList<String> keywordsList) {
        //暴力搜索
        ArrayList<String> result = new ArrayList<>();
        for(String key:keywordsList){
            if(key.matches("(.*)"+prex+"(.*)")){
                result.add(key);
            }
        }
        return result;
    }

    private ArrayList<String> searchAll(String prex, String match){
//        System.out.println(match);
        Set<String> keys = this.keywords.keySet();
        ArrayList<String> res = new ArrayList<>();
        for(String key:keys){
//            System.out.println("HashKey:"+key+", match:"+ match);
            if(key.matches(match)){
//                System.out.println(true);
                res.addAll(this.search(prex, this.keywords.get(key)));
            }
        }
        return res;
    }

    private static void test2(){
        BufferedReader br = null;
        try{
            br = new BufferedReader(new FileReader(Constants.ROOT_PATH + "\\词性序列.txt"));
        }catch(FileNotFoundException e){
            e.printStackTrace();
        }
        String line = null;
        try {
            while ((line = br.readLine()) != null) {
                if (!line.isEmpty()) {
                    String[] tokens = line.split(" ");
                    String[] tags = new String[tokens.length-1];
                    for(int i=0;i<tags.length;++i){
                        tags[i] = tokens[i];
                    }
                    String[] locs =tokens[tokens.length-1].split(",");
                    int begin = Integer.valueOf(locs[0]);
                    int end = Integer.valueOf(locs[1]);
                    String[] keyTags = new String[end-begin+1];
                    int count = 0;
                    for(int i=begin;i<=end;++i){
                        keyTags[count++] = tags[i];
                    }
                    for(String tag:keyTags){
                        System.out.print(tag + " ");
                    }
                    System.out.println("");
                }
            }
        }catch(IOException e){
            e.printStackTrace();
        }
    }
    private static void testSearchPattern(){
        String[] vpTags = new String[]{
                "TTS","TTS"
        };
        String[] tags = new String[]{
                "VN","TP","VP","TTS"
        };
        ArrayList<Span> spanList = new ArrayList<>();
        searchPattern(vpTags,tags,spanList);
        for(Span s:spanList){
            System.out.println(s.getStart() + "," + s.getEnd());
        }
    }

    @Override
    public void testKeywordSearch(POSTaggerME tagger) {
        String preString = "for vehicle to";
        String prex = "d";
        String[] preWords = preString.split(" ");
        String[] tags = tagger.tag(preWords);
        String[] predictTags = this.predictWordTag(preWords, tags);
        ArrayList<String> res = keywordSearch(predictTags, prex);
        System.out.println("上文单词："+preString);
        System.out.println("搜索前缀："+ prex);
        System.out.println("搜索结果为：");
        this.printKeywords(res);
        System.out.println();
    }

    public static void main(String[] args )
    {
//        KeyWordImpl.testSearchPattern();
        String Str = new String("www.ydrtfcfyvg.com");

        System.out.print("返回值 :" );
        System.out.println(Str.matches("(.*)tfc(.*)"));
    }
}
