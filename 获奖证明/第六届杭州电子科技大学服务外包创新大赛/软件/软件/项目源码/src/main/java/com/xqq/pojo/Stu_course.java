package com.xqq.pojo;

import java.io.Serializable;
import java.util.Date;

public class Stu_course extends Stu_courseKey implements Serializable {
    private Date chooseTime;

    private Double examScore;

    private Double allScore;

    private Double commentScore;

    private Double testScore;

    private String oneAnswerSrc;

    private String twoAnswerSrc;

    private String threeAnswerSrc;

    private String fourAnswerSrc;

    private String fiveAnswerSrc;

    private static final long serialVersionUID = 1L;

    public Stu_course(Integer courseId,Integer studentId,Double examScore, Date chooseTime, String fiveAnswerSrc, String fourAnswerSrc, String threeAnswerSrc, String twoAnswerSrc, String oneAnswerSrc, Double allScore, Double commentScore, Double testScore) {
        super(courseId,studentId);
        this.examScore=examScore;
        this.chooseTime=chooseTime;
        this.fiveAnswerSrc=fiveAnswerSrc;
        this.fourAnswerSrc=fourAnswerSrc;
        this.threeAnswerSrc=threeAnswerSrc;
        this.twoAnswerSrc=twoAnswerSrc;
        this.oneAnswerSrc=oneAnswerSrc;
        this.allScore=allScore;
        this.testScore=testScore;
        this.commentScore=commentScore;

    }

    public Stu_course() {

    }

    public Date getChooseTime() {
        return chooseTime;
    }

    public void setChooseTime(Date chooseTime) {
        this.chooseTime = chooseTime;
    }

    public Double getExamScore() {
        return examScore;
    }

    public void setExamScore(Double examScore) {
        this.examScore = examScore;
    }

    public Double getAllScore() {
        return allScore;
    }

    public void setAllScore(Double allScore) {
        this.allScore = allScore;
    }

    public Double getCommentScore() {
        return commentScore;
    }

    public void setCommentScore(Double commentScore) {
        this.commentScore = commentScore;
    }

    public Double getTestScore() {
        return testScore;
    }

    public void setTestScore(Double testScore) {
        this.testScore = testScore;
    }

    public String getOneAnswerSrc() {
        return oneAnswerSrc;
    }

    public void setOneAnswerSrc(String oneAnswerSrc) {
        this.oneAnswerSrc = oneAnswerSrc == null ? null : oneAnswerSrc.trim();
    }

    public String getTwoAnswerSrc() {
        return twoAnswerSrc;
    }

    public void setTwoAnswerSrc(String twoAnswerSrc) {
        this.twoAnswerSrc = twoAnswerSrc == null ? null : twoAnswerSrc.trim();
    }

    public String getThreeAnswerSrc() {
        return threeAnswerSrc;
    }

    public void setThreeAnswerSrc(String threeAnswerSrc) {
        this.threeAnswerSrc = threeAnswerSrc == null ? null : threeAnswerSrc.trim();
    }

    public String getFourAnswerSrc() {
        return fourAnswerSrc;
    }

    public void setFourAnswerSrc(String fourAnswerSrc) {
        this.fourAnswerSrc = fourAnswerSrc == null ? null : fourAnswerSrc.trim();
    }

    public String getFiveAnswerSrc() {
        return fiveAnswerSrc;
    }

    public void setFiveAnswerSrc(String fiveAnswerSrc) {
        this.fiveAnswerSrc = fiveAnswerSrc == null ? null : fiveAnswerSrc.trim();
    }
}