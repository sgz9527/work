package com.xqq.dao;

import com.xqq.pojo.Util;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("utilDao")
public interface UtilMapper {
    int deleteByPrimaryKey(Integer utilId);

    int insert(Util record);

    int insertSelective(Util record);

    Util selectByPrimaryKey(Integer utilId);

    ArrayList<Util> selectByChapterId(Integer chapterId);

    int updateByPrimaryKeySelective(Util record);

    int updateByPrimaryKey(Util record);
}