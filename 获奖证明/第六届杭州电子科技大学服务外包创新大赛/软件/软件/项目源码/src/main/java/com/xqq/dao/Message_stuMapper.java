package com.xqq.dao;

import com.xqq.pojo.Message_stuKey;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface Message_stuMapper {
    int deleteByPrimaryKey(Message_stuKey key);

    int insert(Message_stuKey record);

    int insertSelective(Message_stuKey record);

    Message_stuKey selectByPrimaryKey(@Param("messageId") Integer messageId,@Param("studentId") Integer studentId);
}