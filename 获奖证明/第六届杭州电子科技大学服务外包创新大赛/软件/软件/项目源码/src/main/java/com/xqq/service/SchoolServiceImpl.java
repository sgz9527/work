package com.xqq.service;

import com.xqq.dao.SchorcomMapper;
import com.xqq.pojo.SchOrCom;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

@Service
public class SchoolServiceImpl implements SchoolService {
    @Resource
    SchorcomMapper schoolDao;

    public ArrayList<HashMap<String, Object>> getIndexSch(Integer from, Integer to) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<HashMap<String, Object>>();
        List<SchOrCom> schList = schoolDao.selectAll(from, to);
        mapList = this.toHashMapList(schList);
        return mapList;
    }

    public ArrayList<HashMap<String, Object>> getIndexSch1() {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<HashMap<String, Object>>();
        List<SchOrCom> schList = schoolDao.selectAll1();
        mapList = this.toHashMapList(schList);
        return mapList;
    }

    public ArrayList<HashMap<String, Object>> toHashMapList(List<SchOrCom> schList) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<HashMap<String, Object>>();
        for (int i = 0; i < schList.size(); i++) {
            HashMap<String, Object> map1 = new HashMap<String, Object>();
            map1.put("schoolname", schList.get(i).getComName());
            map1.put("schoolid", schList.get(i).getComId());
            map1.put("type", schList.get(i).getType());
            map1.put("pic", schList.get(i).getComLogo());
            mapList.add(map1);
        }
        return mapList;
    }
}
