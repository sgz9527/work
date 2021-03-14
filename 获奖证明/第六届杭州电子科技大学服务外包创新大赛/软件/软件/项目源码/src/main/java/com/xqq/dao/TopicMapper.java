package com.xqq.dao;

import com.xqq.pojo.Topic;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("topicDao")
public interface TopicMapper {
    int deleteByPrimaryKey(Integer topicId);

    int insert(Topic record);

    int insertSelective(Topic record);

    Topic selectByPrimaryKey(Integer topicId);

    ArrayList<Topic> selectByCourseId(Integer courseId);

    ArrayList<Topic> selectByStudentId(@Param("studentId") Integer studentId);

    int updateByPrimaryKeySelective(Topic record);

    int updateReportNum(@Param("topicId") Integer topicId);

    int updateByPrimaryKey(Topic record);
}