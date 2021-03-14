package com.xqq.dao;

import com.xqq.pojo.Stu_course;
import com.xqq.pojo.Stu_courseKey;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
@Repository("stu_courseDao")
public interface Stu_courseMapper {
    int deleteByPrimaryKey(Stu_courseKey key);

    Stu_course selectByPrimaryKey(Stu_courseKey key);

    List<Stu_course> selectByCourseId(@Param("courseId") Integer courseId);

    int insert(Stu_course record);

    Stu_course selectByStudentIdAndCourseId(@Param("studentId") Integer studentId, @Param("courseId") Integer courseId);

    int insertSelective(Stu_course record);

    List<Stu_course> selectByStudentId(@Param("studentId") Integer studentId);

    int updateByPrimaryKeySelective(Stu_course record);
    int updateByPrimaryKey(Stu_course record);
}