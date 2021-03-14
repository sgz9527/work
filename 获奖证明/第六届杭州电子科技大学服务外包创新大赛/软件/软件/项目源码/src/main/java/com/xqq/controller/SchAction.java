package com.xqq.controller;

import com.xqq.service.SchoolService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

@Controller("SchAction")
public class SchAction {
    ArrayList<HashMap<String, Object>> mapList = new ArrayList<HashMap<String, Object>>();
    @Autowired
    SchoolService schoolServ;

    @RequestMapping("indexSch")
    @ResponseBody
    public List<HashMap<String, Object>> getIndexSch() {
        mapList = schoolServ.getIndexSch1();
        return mapList;
    }
}
