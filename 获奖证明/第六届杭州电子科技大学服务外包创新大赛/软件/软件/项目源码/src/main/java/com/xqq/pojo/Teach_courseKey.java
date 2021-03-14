package com.xqq.pojo;

public class Teach_courseKey {
    private Byte examAgree;
    private Integer teacherId;
    private Integer courseId;
    private String examArgumentSuplement;

    private static final long serialVersionUID = 1L;

    public Byte getExamAgree() {
        return examAgree;
    }

    public void setExamAgree(Byte examAgree) {
        this.examAgree = examAgree;
    }

    public String getExamArgumentSuplement() {
        return examArgumentSuplement;
    }

    public void setExamArgumentSuplement(String examArgumentSuplement) {
        this.examArgumentSuplement = examArgumentSuplement == null ? null : examArgumentSuplement.trim();
    }

    public Integer getTeacherId() {
        return teacherId;
    }

    public void setTeacherId(Integer teacherId) {
        this.teacherId = teacherId;
    }

    public Integer getCourseId() {
        return courseId;
    }

    public void setCourseId(Integer courseId) {
        this.courseId = courseId;
    }
}