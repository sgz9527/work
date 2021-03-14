package com.xqq.dao;

import com.xqq.pojo.Teach_courseKey;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
@Repository("teach_courseDao")
public interface Teach_courseMapper {
    int deleteByPrimaryKey(Teach_courseKey key);

    int insert(Teach_courseKey record);

    int insertSelective(Teach_courseKey record);

    List<Teach_courseKey> selectByCourseId(@Param("courseid") Integer courseid);

    Teach_courseKey selectByTeacherIdAndCourseId(@Param("teacherId") Integer teacherId, @Param("courseId") Integer courseId);

    ArrayList<Teach_courseKey> selectByTeacherId(Integer teacherId);
    Teach_courseKey selectByPrimaryKey(@Param("teacherId")Integer teacherId,@Param("courseId")Integer courseId);

    int updateByPrimaryKeySelective(Teach_courseKey tc);
}