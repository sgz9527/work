package com.xqq.service;

import java.util.ArrayList;
import java.util.HashMap;

public interface SchoolService {
    /**
     * 获取首页院校展示
     * @param from
     * @param to
     * @return
     */
    public ArrayList<HashMap<String, Object>> getIndexSch(Integer from, Integer to);

    /**
     * 获取所有院校信息
     * @return
     */

    public ArrayList<HashMap<String, Object>> getIndexSch1();
}
