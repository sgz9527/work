package com.xqq.dao;

import com.xqq.pojo.Training;
import org.springframework.stereotype.Repository;

@Repository("trainingDao")
public interface TrainingMapper {
    int deleteByPrimaryKey(Integer id);

    int insert(Training record);

    int insertSelective(Training record);

    Training selectByPrimaryKey(Integer id);

    int updateByPrimaryKeySelective(Training record);

    int updateByPrimaryKey(Training record);
}