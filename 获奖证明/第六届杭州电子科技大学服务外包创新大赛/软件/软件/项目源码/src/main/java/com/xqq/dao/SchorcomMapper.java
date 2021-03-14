package com.xqq.dao;

import com.xqq.pojo.SchOrCom;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository("schoolDao")
public interface SchorcomMapper {
    int deleteByPrimaryKey(Integer comId);

    int insert(SchOrCom record);

    int insertSelective(SchOrCom record);

    SchOrCom selectByPrimaryKey(Integer comId);

    List<SchOrCom> selectAll(@Param("from") Integer from, @Param("to") Integer to);

    List<SchOrCom> selectAll1();

    int updateByPrimaryKeySelective(SchOrCom record);

    int updateByPrimaryKey(SchOrCom record);
}