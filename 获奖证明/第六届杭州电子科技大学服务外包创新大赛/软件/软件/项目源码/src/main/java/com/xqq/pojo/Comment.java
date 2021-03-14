package com.xqq.pojo;

import java.io.Serializable;

public class Comment implements Serializable {
    private Integer commentId;

    private String commentTime;

    private String commentInfo;

    private Integer commentWritterId;

    private Short commentWritterType;

    private Integer belongTopicId;

    private Integer replyToId;
    private Integer reportNum;
    private Integer isForbiden;
    private Topic topic;
    private static final long serialVersionUID = 1L;

    public Integer getCommentId() {
        return commentId;
    }

    public void setCommentId(Integer commentId) {
        this.commentId = commentId;
    }

    public String getCommentTime() {
        return commentTime;
    }

    public void setCommentTime(String commentTime) {
        this.commentTime = commentTime == null ? null : commentTime.trim();
    }

    public String getCommentInfo() {
        return commentInfo;
    }

    public void setCommentInfo(String commentInfo) {
        this.commentInfo = commentInfo == null ? null : commentInfo.trim();
    }

    public Integer getCommentWritterId() {
        return commentWritterId;
    }

    public void setCommentWritterId(Integer commentWritterId) {
        this.commentWritterId = commentWritterId;
    }

    public Short getCommentWritterType() {
        return commentWritterType;
    }

    public void setCommentWritterType(Short commentWritterType) {
        this.commentWritterType = commentWritterType;
    }

    public Integer getReplyToId() {
        return replyToId;
    }

    public void setReplyToId(Integer replyToId) {
        this.replyToId = replyToId;
    }

    public Integer getBelongTopicId() {
        return belongTopicId;
    }

    public void setBelongTopicId(Integer belongTopicId) {
        this.belongTopicId = belongTopicId;
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

    public Topic getTopic() {
        return topic;
    }

    public void setTopic(Topic topic) {
        this.topic = topic;
    }
}