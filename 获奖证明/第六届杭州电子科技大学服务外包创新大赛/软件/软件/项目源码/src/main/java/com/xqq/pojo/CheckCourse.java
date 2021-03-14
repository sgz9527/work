package com.xqq.pojo;

import java.io.Serializable;

public class CheckCourse implements Serializable {
    private Integer checkId;

    private String courseName;

    private String courseIntroduce;

    private String introduceSrc;

    private String courseType;

    private Integer teacherId;

    private Integer belongSchId;

    private String checkState;

    private String teacherPhone;

    private String posterSrc;

    private String suplement;

    private static final long serialVersionUID = 1L;

    public Integer getCheckId() {
        return checkId;
    }

    public void setCheckId(Integer checkId) {
        this.checkId = checkId;
    }

    public String getCourseName() {
        return courseName;
    }

    public void setCourseName(String courseName) {
        this.courseName = courseName == null ? null : courseName.trim();
    }

    public String getCourseIntroduce() {
        return courseIntroduce;
    }

    public void setCourseIntroduce(String courseIntroduce) {
        this.courseIntroduce = courseIntroduce == null ? null : courseIntroduce.trim();
    }

    public String getIntroduceSrc() {
        return introduceSrc;
    }

    public void setIntroduceSrc(String introduceSrc) {
        this.introduceSrc = introduceSrc == null ? null : introduceSrc.trim();
    }

    public String getCourseType() {
        return courseType;
    }

    public void setCourseType(String courseType) {
        this.courseType = courseType == null ? null : courseType.trim();
    }

    public Integer getTeacherId() {
        return teacherId;
    }

    public void setTeacherId(Integer teacherId) {
        this.teacherId = teacherId;
    }

    public Integer getBelongSchId() {
        return belongSchId;
    }

    public void setBelongSchId(Integer belongSchId) {
        this.belongSchId = belongSchId;
    }

    public String getCheckState() {
        return checkState;
    }

    public void setCheckState(String checkState) {
        this.checkState = checkState == null ? null : checkState.trim();
    }

    public String getTeacherPhone() {
        return teacherPhone;
    }

    public void setTeacherPhone(String teacherPhone) {
        this.teacherPhone = teacherPhone == null ? null : teacherPhone.trim();
    }

    public String getPosterSrc() {
        return posterSrc;
    }

    public void setPosterSrc(String posterSrc) {
        this.posterSrc = posterSrc == null ? null : posterSrc.trim();
    }

    public String getSuplement() {
        return suplement;
    }

    public void setSuplement(String suplement) {
        this.suplement = suplement == null ? null : suplement.trim();
    }
}