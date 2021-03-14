package com.xqq.pojo;

import com.xqq.constant.Constants;

import java.io.Serializable;
import java.util.Date;

public class Message implements Serializable {
    private Integer messageId;

    private Short messageType;

    private Integer messageCourseId;

    private Integer messageTopicId;

    private Integer messageCommentId;

    private Integer messageTestId;

    private Integer messageExamId;

    private Integer messageStudentId;

    private Date messageForbidenBeginTime;

    private Date messageForbidenEndTime;

    private Integer messageExcerciseId;

    private Date messageTime;
    private Stu_course stuCourse;
    private Comment comment;

    private static final long serialVersionUID = 1L;

    public Integer getMessageId() {
        return messageId;
    }

    public void setMessageId(Integer messageId) {
        this.messageId = messageId;
    }

    public Short getMessageType() {
        return messageType;
    }

    public void setMessageType(Short messageType) {
        this.messageType = messageType;
    }

    public Integer getMessageCourseId() {
        return messageCourseId;
    }

    public void setMessageCourseId(Integer messageCourseId) {
        this.messageCourseId = messageCourseId;
    }

    public Integer getMessageTopicId() {
        return messageTopicId;
    }

    public void setMessageTopicId(Integer messageTopicId) {
        this.messageTopicId = messageTopicId;
    }

    public Integer getMessageCommentId() {
        return messageCommentId;
    }

    public void setMessageCommentId(Integer messageCommentId) {
        this.messageCommentId = messageCommentId;
    }

    public Integer getMessageTestId() {
        return messageTestId;
    }

    public void setMessageTestId(Integer messageTestId) {
        this.messageTestId = messageTestId;
    }

    public Integer getMessageExamId() {
        return messageExamId;
    }

    public void setMessageExamId(Integer messageExamId) {
        this.messageExamId = messageExamId;
    }

    public Integer getMessageStudentId() {
        if(messageStudentId==null){
            messageStudentId= Constants.NULL_STUDENT_ID;
        }
        return messageStudentId;
    }

    public void setMessageStudentId(Integer messageStudentId) {
        this.messageStudentId = messageStudentId;
    }

    public Date getMessageForbidenBeginTime() {
        return messageForbidenBeginTime;
    }

    public void setMessageForbidenBeginTime(Date messageForbidenBeginTime) {
        this.messageForbidenBeginTime = messageForbidenBeginTime;
    }

    public Date getMessageForbidenEndTime() {
        return messageForbidenEndTime;
    }

    public void setMessageForbidenEndTime(Date messageForbidenEndTime) {
        this.messageForbidenEndTime = messageForbidenEndTime;
    }

    public Integer getMessageExcerciseId() {
        return messageExcerciseId;
    }

    public void setMessageExcerciseId(Integer messageExcerciseId) {
        this.messageExcerciseId = messageExcerciseId;
    }

    public Date getMessageTime() {
        return messageTime;
    }

    public void setMessageTime(Date messageTime) {
        this.messageTime = messageTime;
    }

    public Stu_course getStuCourse() {
        return stuCourse;
    }

    public void setStuCourse(Stu_course stuCourse) {
        this.stuCourse = stuCourse;
    }

    public Comment getComment() {
        return comment;
    }

    public void setComment(Comment comment) {
        this.comment = comment;
    }
}