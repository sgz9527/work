package com.xqq.pojo;

import java.io.Serializable;
import java.util.Date;

public class Course implements Serializable {
    private Integer courseId;

    private String courseName;

    private String courseIntroduce;

    private String coursePostSrc;

    private Integer courseProgress;

    private Integer belongSchId;

    private String courseType;

    private String introduceVideoSrc;

    private String evaluationLevel;

    private Double evaluationExamWeight;

    private Double evaluationChatWeight;

    private Double evaluationTestWeight;

    private Date examStartTime;

    private Date examEndTime;

    private String examSuplement;

    private Double examLimitTime;

    private String examOneTitle;

    private String examOneAnswer;

    private String examTwoTitle;

    private String examTwoAnswer;

    private String examThreeTitle;

    private String examThreeAnswer;

    private String examFourTitle;

    private String examFourAnswer;

    private String examFiveTitle;

    private String examFiveAnswer;

    private Integer examPublishTeacherId;
    private byte examPass;

    private static final long serialVersionUID = 1L;

    public Integer getCourseId() {
        return courseId;
    }

    public void setCourseId(Integer courseId) {
        this.courseId = courseId;
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

    public String getCoursePostSrc() {
        return coursePostSrc;
    }

    public void setCoursePostSrc(String coursePostSrc) {
        this.coursePostSrc = coursePostSrc == null ? null : coursePostSrc.trim();
    }

    public Integer getCourseProgress() {
        return courseProgress;
    }

    public void setCourseProgress(Integer courseProgress) {
        this.courseProgress = courseProgress;
    }

    public Integer getBelongSchId() {
        return belongSchId;
    }

    public void setBelongSchId(Integer belongSchId) {
        this.belongSchId = belongSchId;
    }

    public String getCourseType() {
        return courseType;
    }

    public void setCourseType(String courseType) {
        this.courseType = courseType == null ? null : courseType.trim();
    }

    public String getIntroduceVideoSrc() {
        return introduceVideoSrc;
    }

    public void setIntroduceVideoSrc(String introduceVideoSrc) {
        this.introduceVideoSrc = introduceVideoSrc == null ? null : introduceVideoSrc.trim();
    }

    public String getEvaluationLevel() {
        return evaluationLevel;
    }

    public void setEvaluationLevel(String evaluationLevel) {
        this.evaluationLevel = evaluationLevel == null ? null : evaluationLevel.trim();
    }

    public Double getEvaluationExamWeight() {
        return evaluationExamWeight;
    }

    public void setEvaluationExamWeight(Double evaluationExamWeight) {
        this.evaluationExamWeight = evaluationExamWeight;
    }

    public Double getEvaluationChatWeight() {
        return evaluationChatWeight;
    }

    public void setEvaluationChatWeight(Double evaluationChatWeight) {
        this.evaluationChatWeight = evaluationChatWeight;
    }

    public Double getEvaluationTestWeight() {
        return evaluationTestWeight;
    }

    public void setEvaluationTestWeight(Double evaluationTestWeight) {
        this.evaluationTestWeight = evaluationTestWeight;
    }

    public Date getExamStartTime() {
        return examStartTime;
    }

    public void setExamStartTime(Date examStartTime) {
        this.examStartTime = examStartTime;
    }

    public Date getExamEndTime() {
        return examEndTime;
    }

    public void setExamEndTime(Date examEndTime) {
        this.examEndTime = examEndTime;
    }

    public String getExamSuplement() {
        return examSuplement;
    }

    public void setExamSuplement(String examSuplement) {
        this.examSuplement = examSuplement == null ? null : examSuplement.trim();
    }

    public Double getExamLimitTime() {
        return examLimitTime;
    }

    public void setExamLimitTime(Double examLimitTime) {
        this.examLimitTime = examLimitTime;
    }

    public String getExamOneTitle() {
        return examOneTitle;
    }

    public void setExamOneTitle(String examOneTitle) {
        this.examOneTitle = examOneTitle == null ? null : examOneTitle.trim();
    }

    public String getExamOneAnswer() {
        return examOneAnswer;
    }

    public void setExamOneAnswer(String examOneAnswer) {
        this.examOneAnswer = examOneAnswer == null ? null : examOneAnswer.trim();
    }

    public String getExamTwoTitle() {
        return examTwoTitle;
    }

    public void setExamTwoTitle(String examTwoTitle) {
        this.examTwoTitle = examTwoTitle == null ? null : examTwoTitle.trim();
    }

    public String getExamTwoAnswer() {
        return examTwoAnswer;
    }

    public void setExamTwoAnswer(String examTwoAnswer) {
        this.examTwoAnswer = examTwoAnswer == null ? null : examTwoAnswer.trim();
    }

    public String getExamThreeTitle() {
        return examThreeTitle;
    }

    public void setExamThreeTitle(String examThreeTitle) {
        this.examThreeTitle = examThreeTitle == null ? null : examThreeTitle.trim();
    }

    public String getExamThreeAnswer() {
        return examThreeAnswer;
    }

    public void setExamThreeAnswer(String examThreeAnswer) {
        this.examThreeAnswer = examThreeAnswer == null ? null : examThreeAnswer.trim();
    }

    public String getExamFourTitle() {
        return examFourTitle;
    }

    public void setExamFourTitle(String examFourTitle) {
        this.examFourTitle = examFourTitle == null ? null : examFourTitle.trim();
    }

    public String getExamFourAnswer() {
        return examFourAnswer;
    }

    public void setExamFourAnswer(String examFourAnswer) {
        this.examFourAnswer = examFourAnswer == null ? null : examFourAnswer.trim();
    }

    public String getExamFiveTitle() {
        return examFiveTitle;
    }

    public void setExamFiveTitle(String examFiveTitle) {
        this.examFiveTitle = examFiveTitle == null ? null : examFiveTitle.trim();
    }

    public String getExamFiveAnswer() {
        return examFiveAnswer;
    }

    public void setExamFiveAnswer(String examFiveAnswer) {
        this.examFiveAnswer = examFiveAnswer == null ? null : examFiveAnswer.trim();
    }

    public Integer getExamPublishTeacherId() {
        return examPublishTeacherId;
    }

    public void setExamPublishTeacherId(Integer examPublishTeacherId) {
        this.examPublishTeacherId = examPublishTeacherId;
    }

    public byte getExamPass() {
        return examPass;
    }

    public void setExamPass(byte examPass) {
        this.examPass = examPass;
    }
}