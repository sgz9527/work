package com.xqq.dao;

import com.xqq.pojo.Course;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
@Repository("courseDao")
public interface CourseMapper {
    int deleteByPrimaryKey(@Param("courseId") Integer courseId);

    int insert(Course record);

    int insertSelective(Course record);

    Course selectByPrimaryKey(@Param("courseId") Integer courseId);

    int updateByPrimaryKeySelective(Course record);

    int updateByPrimaryKey(Course record);

    List<Course> selectTop12();

    List<Course> selectAll();

    List<Course> selectByFuzzy(@Param("key") String key);

    List<Course> selectCourseByType(@Param("type") String type);
}