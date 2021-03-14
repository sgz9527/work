package com.xqq.dao;

import com.xqq.pojo.Praise_commentKey;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository("Praise_commentMapper")
public interface Praise_commentMapper {
    int deleteByPrimaryKey(Praise_commentKey key);

    int insert(Praise_commentKey record);

    int insertSelective(Praise_commentKey record);

    List<Praise_commentKey> selectByCommentId(@Param("commentId") Integer commentId);

    Praise_commentKey selectByPrimaryKey(Praise_commentKey pra);
}