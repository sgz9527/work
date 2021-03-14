/*
 * Copyright (C) 2017 Baidu, Inc. All Rights Reserved.
 */
package com.xqq.staticmethod;

import com.fasterxml.jackson.core.JsonParseException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.xqq.pojo.Stu_course;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.lang.reflect.Type;
import java.util.ArrayList;


/**
 * Json工具类.
 */
public class GsonUtils {
    private static Gson gson = new GsonBuilder().create();

    public static String toJson(Object value) {
        return gson.toJson(value);
    }

    public static <T> T fromJson(String json, Class<T> classOfT) throws JsonParseException {
        return gson.fromJson(json, classOfT);
    }

    public static <T> T fromJson(String json, Type typeOfT) throws JsonParseException {
        return (T) gson.fromJson(json, typeOfT);
    }
    public static JSONArray classListToJSONArray(ArrayList<Stu_course> srcList){
        JSONArray jsonArray=new JSONArray();
        for(int i=0;i<srcList.size();i++){
            String jsonStr=new Gson().toJson(srcList.get(i));
            JSONObject jsob=null;
            try {
                jsob=(JSONObject)new JSONParser().parse(jsonStr);
            } catch (ParseException e) {
                e.printStackTrace();
            }
            jsonArray.add(jsob);
        }
        return jsonArray;
    }
    public static JSONArray objectListToJSONArray(ArrayList<Object> srcList){
        JSONArray jsonArray=new JSONArray();
        for(int i=0;i<srcList.size();i++){
            JSONObject jsob=null;
            String jsonStr=new Gson().toJson(srcList.get(i));
            try {
                jsob=(JSONObject)new JSONParser().parse(jsonStr);
            } catch (ParseException e) {
                e.printStackTrace();
            }
            jsonArray.add(jsob);
        }
        return jsonArray;
    }
}
