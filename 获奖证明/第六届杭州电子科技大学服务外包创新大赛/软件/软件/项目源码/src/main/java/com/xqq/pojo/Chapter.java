package com.xqq.pojo;

import java.io.Serializable;

public class Chapter implements Serializable {
    private Integer chapterId;

    private String chapterTitle;

    private String chapterName;

    private Integer belongCourseId;
    private Course course;
    private Integer chapterOrder;
    private static final long serialVersionUID = 1L;

    public Integer getChapterId() {
        return chapterId;
    }

    public void setChapterId(Integer chapterId) {
        this.chapterId = chapterId;
    }

    public String getChapterTitle() {
        return chapterTitle;
    }

    public void setChapterTitle(String chapterTitle) {
        this.chapterTitle = chapterTitle == null ? null : chapterTitle.trim();
    }

    public String getChapterName() {
        return chapterName;
    }

    public void setChapterName(String chapterName) {
        this.chapterName = chapterName == null ? null : chapterName.trim();
    }

    public Integer getBelongCourseId() {
        return belongCourseId;
    }

    public void setBelongCourseId(Integer belongCourseId) {
        this.belongCourseId = belongCourseId;
    }

    public Course getCourse() {
        return course;
    }

    public void setCourse(Course course) {
        this.course = course;
    }

    public Integer getChapterOrder() {
        return chapterOrder;
    }

    public void setChapterOrder(Integer chapterOrder) {
        this.chapterOrder = chapterOrder;
    }
}