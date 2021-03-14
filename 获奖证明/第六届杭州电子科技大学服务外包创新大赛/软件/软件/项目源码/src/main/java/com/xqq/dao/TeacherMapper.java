package com.xqq.dao;

import com.xqq.pojo.Teacher;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

@Repository("TeacherMapper")
public interface TeacherMapper {
    int deleteByPrimaryKey(Integer teacherId);

    int insert(Teacher record);

    int insertSelective(Teacher record);

    Teacher selectByPrimaryKey(@Param("teacherId") Integer teacherId);

    Teacher selectByAccountAndPassword(@Param("TeacherAccount") String TeacherAccount, @Param("TeacherPassword") String TeacherPassword);

    int updateByPrimaryKeySelective(Teacher record);

    int updateByPrimaryKey(Teacher record);

    Teacher selectByAccount(String teacherAccount);

}