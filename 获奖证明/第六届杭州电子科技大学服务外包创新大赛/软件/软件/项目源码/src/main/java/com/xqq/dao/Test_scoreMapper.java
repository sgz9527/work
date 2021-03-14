package com.xqq.dao;

import com.xqq.pojo.Test_score;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("test_score_Dao")
public interface Test_scoreMapper {
    int deleteByPrimaryKey(Integer scoreId);

    int insert(Test_score record);

    int insertSelective(Test_score record);

    Test_score selectByPrimaryKey(Integer scoreId);

    Test_score selectByStudentIdAndTestId(@Param("studentId") Integer studentId, @Param("testId") Integer testId);

    int updateByPrimaryKeySelective(Test_score record);

    int updateAlreadyCount(@Param("studentId") Integer studentId, @Param("testId") Integer testId);

    int updateScore(@Param("score") Float score, @Param("studentId") Integer studentId, @Param("testId") Integer testId);

    int updateByPrimaryKey(Test_score record);

    int deleteByStudentIdAndCourseId(@Param("studentId") Integer studentId, @Param("courseId") Integer courseId);

    ArrayList<Test_score> selectByStudentIdAndCourseId(@Param("studentId") Integer studentId, @Param("courseId") Integer courseId);

    ArrayList<Test_score> selectByTestId(Integer testId);
}