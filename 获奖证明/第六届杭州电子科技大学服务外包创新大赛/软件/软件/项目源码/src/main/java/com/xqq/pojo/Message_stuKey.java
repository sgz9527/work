package com.xqq.pojo;

import java.io.Serializable;

public class Message_stuKey implements Serializable {
    private Integer messageId;

    private Integer studentId;

    private static final long serialVersionUID = 1L;

    public Integer getMessageId() {
        return messageId;
    }

    public void setMessageId(Integer messageId) {
        this.messageId = messageId;
    }

    public Integer getStudentId() {
        return studentId;
    }

    public void setStudentId(Integer studentId) {
        this.studentId = studentId;
    }
}