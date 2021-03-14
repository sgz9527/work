package com.xqq.pojo;

import java.io.Serializable;

/**
 * @author xqq
 */
public class StuCourse extends Stu_course implements Serializable {
    private String courseName;
    private String studentName;
    private String studentAccount;
    public StuCourse(Stu_course stuc){
        super(stuc.getCourseId(),stuc.getStudentId(),stuc.getExamScore(),stuc.getChooseTime(),stuc.getFiveAnswerSrc(),stuc.getFourAnswerSrc(),stuc.getThreeAnswerSrc(),stuc.getTwoAnswerSrc(),stuc.getOneAnswerSrc(),stuc.getAllScore(),stuc.getCommentScore(),stuc.getTestScore());
        this.courseName="";
        this.studentName="";
        this.studentAccount="";
    }
    public String getCourseName() {
        return courseName;
    }

    public void setCourseName(String courseName) {
        this.courseName = courseName;
    }

    public String getStudentName() {
        return studentName;
    }

    public void setStudentName(String studentName) {
        this.studentName = studentName;
    }

    public String getStudentAccount() {
        return studentAccount;
    }

    public void setStudentAccount(String studentAccount) {
        this.studentAccount = studentAccount;
    }
}
