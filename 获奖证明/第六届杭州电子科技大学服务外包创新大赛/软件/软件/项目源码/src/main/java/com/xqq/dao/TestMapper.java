package com.xqq.dao;

import com.xqq.pojo.Test;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("testDao")
public interface TestMapper {
    int deleteByPrimaryKey(Integer testId);

    int insert(Test record);

    int insertSelective(Test record);

    Test selectByPrimaryKey(Integer testId);

    ArrayList<Test> selectByChapterId(@Param("chapterId") Integer chapterId);

    ArrayList<Test> selectByChapterIdWait(@Param("chapterId") Integer chapterId);

    int updateByPrimaryKeySelective(Test record);

    int updateByPrimaryKey(Test record);
}