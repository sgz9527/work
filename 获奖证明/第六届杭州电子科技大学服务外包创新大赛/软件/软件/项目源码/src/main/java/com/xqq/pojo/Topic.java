package com.xqq.pojo;

import java.io.Serializable;

public class Topic implements Serializable {
    private Integer topicId;

    private Integer belongCourseId;

    private String topicDetail;

    private String topicTime;

    private Integer topicWritterId;

    private Short topicWritterType;
    private Integer reportNum;
    private Integer isForbiden;

    private static final long serialVersionUID = 1L;

    public Integer getTopicId() {
        return topicId;
    }

    public void setTopicId(Integer topicId) {
        this.topicId = topicId;
    }

    public Integer getBelongCourseId() {
        return belongCourseId;
    }

    public void setBelongCourseId(Integer belongCourseId) {
        this.belongCourseId = belongCourseId;
    }

    public String getTopicDetail() {
        return topicDetail;
    }

    public void setTopicDetail(String topicDetail) {
        this.topicDetail = topicDetail == null ? null : topicDetail.trim();
    }

    public String getTopicTime() {
        return topicTime;
    }

    public void setTopicTime(String topicTime) {
        this.topicTime = topicTime == null ? null : topicTime.trim();
    }

    public Integer getTopicWritterId() {
        return topicWritterId;
    }

    public void setTopicWritterId(Integer topicWritterId) {
        this.topicWritterId = topicWritterId;
    }

    public Short getTopicWritterType() {
        return topicWritterType;
    }

    public void setTopicWritterType(Short topicWritterType) {
        this.topicWritterType = topicWritterType;
    }

    public Integer getReportNum() {
        return reportNum;
    }

    public void setReportNum(Integer reportNum) {
        this.reportNum = reportNum;
    }

    public Integer getIsForbiden() {
        return isForbiden;
    }

    public void setIsForbiden(Integer isForbiden) {
        this.isForbiden = isForbiden;
    }
}