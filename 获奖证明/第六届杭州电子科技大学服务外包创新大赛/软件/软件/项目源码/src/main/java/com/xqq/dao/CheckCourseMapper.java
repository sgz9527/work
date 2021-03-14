package com.xqq.dao;

import com.xqq.pojo.CheckCourse;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;

@Repository("checkCourseDao")
public interface CheckCourseMapper {
    int deleteByPrimaryKey(Integer checkId);

    int insert(CheckCourse record);

    int insertSelective(CheckCourse record);

    CheckCourse selectByPrimaryKey(Integer checkId);

    int updateByPrimaryKeySelective(CheckCourse record);

    int updateByPrimaryKey(CheckCourse record);

    ArrayList<CheckCourse> selectByTeacherId(Integer teacherId);

    List<CheckCourse> selectByCheckState();
}