package com.xqq.dao;

import com.xqq.pojo.TestProblem;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("TestProblemMapper")
public interface TestProblemMapper {
    int deleteByPrimaryKey(Integer testProblemId);

    int insert(TestProblem record);

    int insertSelective(TestProblem record);

    TestProblem selectByPrimaryKey(Integer testProblemId);

    ArrayList<TestProblem> selectByTestId(@Param("testId") Integer testId);

    int updateByPrimaryKeySelective(TestProblem record);

    int updateByPrimaryKey(TestProblem record);
}