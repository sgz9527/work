package com.xqq.dao;

import com.xqq.pojo.Message;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
@Repository("messageDao")
public interface MessageMapper {
    int deleteByPrimaryKey(Integer messageId);

    int insert(Message record);

    int insertSelective(Message record);

    Message selectByPrimaryKey(Integer messageId);

    int updateByPrimaryKeySelective(Message record);

    int updateByPrimaryKey(Message record);
    ArrayList<Message> selectByStudentId(Integer studentId);

    Message selectByMessageTypeAndTestId(@Param("messageType")Short messageType, @Param("testId")Integer testId);

    void deleteByTime();

    ArrayList<Message> selectByMessageTypeAndStudentId(Integer studentId);

    Message selectByMessageTypeAndExamId(@Param("messageType")Short messageType, @Param("examId")Integer messageExamId);

}