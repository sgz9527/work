package com.xqq.pojo;

import java.io.Serializable;

public class Praise_commentKey implements Serializable {
    private Integer commentId;

    private Integer praiseUserId;

    private Short praiseUserType;

    private static final long serialVersionUID = 1L;

    public Integer getCommentId() {
        return commentId;
    }

    public void setCommentId(Integer commentId) {
        this.commentId = commentId;
    }

    public Integer getPraiseUserId() {
        return praiseUserId;
    }

    public void setPraiseUserId(Integer praiseUserId) {
        this.praiseUserId = praiseUserId;
    }

    public Short getPraiseUserType() {
        return praiseUserType;
    }

    public void setPraiseUserType(Short praiseUserType) {
        this.praiseUserType = praiseUserType;
    }
}