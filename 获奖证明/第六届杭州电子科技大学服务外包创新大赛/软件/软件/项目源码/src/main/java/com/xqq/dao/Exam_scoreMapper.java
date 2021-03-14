package com.xqq.dao;

import com.xqq.pojo.Exam_score;
import org.springframework.stereotype.Repository;

@Repository("Exam_scoreMapper")
public interface Exam_scoreMapper {
    int deleteByPrimaryKey(Integer examId);

    int insert(Exam_score record);

    int insertSelective(Exam_score record);

    Exam_score selectByPrimaryKey(Integer examId);

    int updateByPrimaryKeySelective(Exam_score record);

    int updateByPrimaryKey(Exam_score record);
}