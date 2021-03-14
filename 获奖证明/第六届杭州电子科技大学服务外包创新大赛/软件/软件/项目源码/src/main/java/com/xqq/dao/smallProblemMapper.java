package com.xqq.dao;

import com.xqq.pojo.smallProblem;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("smallProblemMapper")
public interface smallProblemMapper {
    int deleteByPrimaryKey(Integer smallProblemId);

    int insert(smallProblem record);

    int insertSelective(smallProblem record);

    smallProblem selectByPrimaryKey(Integer smallProblemId);

    ArrayList<smallProblem> selectByTestProblemId(@Param("testProblemId") Integer testProblemId);

    int updateByPrimaryKeySelective(smallProblem record);

    int updateByPrimaryKey(smallProblem record);
}