package com.xqq.pojo;

import java.io.Serializable;

public class Stu_courseKey implements Serializable {
    private Integer studentId;

    private Integer courseId;

    private static final long serialVersionUID = 1L;

    public Stu_courseKey(Integer courseId, Integer studentId) {
        this.courseId=courseId;
        this.studentId=studentId;
    }
    public Stu_courseKey(){

    }

    public Integer getStudentId() {
        return studentId;
    }

    public void setStudentId(Integer studentId) {
        this.studentId = studentId;
    }

    public Integer getCourseId() {
        return courseId;
    }

    public void setCourseId(Integer courseId) {
        this.courseId = courseId;
    }
}