package com.xqq.dao;

import com.xqq.pojo.Student;
import org.springframework.stereotype.Repository;

@Repository("studentDao")
public interface StudentMapper {
    int deleteByPrimaryKey(Integer studentId);

    int insert(Student record);

    int insertSelective(Student record);

    Student selectByPrimaryKey(Integer studentId);

    int updateByPrimaryKeySelective(Student record);

    int updateByPrimaryKey(Student record);

    int updateBreakComRule(Integer studentId);

}