package com.xqq.dao;

import com.xqq.pojo.Source;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("sourceDao")
public interface SourceMapper {
    int deleteByPrimaryKey(Integer sourceId);

    int insert(Source record);

    int insertSelective(Source record);

    Source selectByPrimaryKey(Integer sourceId);

    ArrayList<Source> selectByUtilId(@Param("utilId") Integer utilId);

    ArrayList<Source> selectByUtilIdAndType(@Param("utilId") Integer utilId, @Param("sourceType") String sourceType);

    int updateByPrimaryKeySelective(Source record);

    int updateByPrimaryKey(Source record);
}