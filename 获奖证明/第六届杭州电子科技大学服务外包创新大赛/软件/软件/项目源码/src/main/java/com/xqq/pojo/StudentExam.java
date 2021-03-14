package com.xqq.pojo;

import com.xqq.staticmethod.getDate;

/**
 * @author xqq
 */
public class StudentExam{
    private String isFinish;
    private String isEnd;
    private String examStartTime;

    private String examEndTime;

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
    private String oneSrc;
    private String twoSrc;
    private String threeSrc;
    private String fourSrc;
    private String fiveSrc;
    private byte examPass;
    public String getIsFinish() {
        return isFinish;
    }
    public StudentExam(Course c,Stu_course stuC){
        String endTime=getDate.getYMD(c.getExamEndTime());
        this.setExamEndTime(endTime);
        this.setExamFiveTitle(c.getExamFiveTitle());
        this.setExamFiveAnswer(c.getExamFiveAnswer());
        this.setExamFourAnswer(c.getExamFourAnswer());
        this.setExamFourTitle(c.getExamFourTitle());
        this.setExamThreeAnswer(c.getExamThreeAnswer());
        this.setExamThreeTitle(c.getExamThreeTitle());
        this.setExamTwoAnswer(c.getExamTwoAnswer());
        this.setExamTwoTitle(c.getExamTwoTitle());
        this.setExamOneAnswer(c.getExamOneAnswer());
        this.setExamOneTitle(c.getExamOneTitle());
        this.setExamPublishTeacherId(c.getExamPublishTeacherId());
        this.setExamSuplement(c.getExamSuplement());
        String startTime= getDate.getYMD(c.getExamStartTime());
        this.setExamStartTime(startTime);
        this.setExamPass(c.getExamPass());
        this.setExamLimitTime(c.getExamLimitTime());
        this.setOneSrc(stuC.getOneAnswerSrc());
        this.setTwoSrc(stuC.getTwoAnswerSrc());
        this.setThreeSrc(stuC.getThreeAnswerSrc());
        this.setFourSrc(stuC.getFourAnswerSrc());
        this.setFiveSrc(stuC.getFiveAnswerSrc());
    }
    public StudentExam(Course c){
        String endTime=getDate.getYMD(c.getExamEndTime());
        this.setExamEndTime(endTime);
        this.setExamFiveTitle(c.getExamFiveTitle());
        this.setExamFiveAnswer(c.getExamFiveAnswer());
        this.setExamFourAnswer(c.getExamFourAnswer());
        this.setExamFourTitle(c.getExamFourTitle());
        this.setExamThreeAnswer(c.getExamThreeAnswer());
        this.setExamThreeTitle(c.getExamThreeTitle());
        this.setExamTwoAnswer(c.getExamTwoAnswer());
        this.setExamTwoTitle(c.getExamTwoTitle());
        this.setExamOneAnswer(c.getExamOneAnswer());
        this.setExamOneTitle(c.getExamOneTitle());
        this.setExamPublishTeacherId(c.getExamPublishTeacherId());
        this.setExamSuplement(c.getExamSuplement());
        String startTime= getDate.getYMD(c.getExamStartTime());
        this.setExamStartTime(startTime);
        this.setExamPass(c.getExamPass());
        this.setExamLimitTime(c.getExamLimitTime());
        this.setOneSrc("");
        this.setTwoSrc("");
        this.setThreeSrc("");
        this.setFourSrc("");
        this.setFiveSrc("");
    }

    public void setIsFinish(String isFinish) {
        this.isFinish = isFinish;
    }

    public String getIsEnd() {
        return isEnd;
    }

    public void setIsEnd(String isEnd) {
        this.isEnd = isEnd;
    }

    public String getExamStartTime() {
        return examStartTime;
    }

    public void setExamStartTime(String examStartTime) {
        this.examStartTime = examStartTime;
    }

    public String getExamEndTime() {
        return examEndTime;
    }

    public void setExamEndTime(String examEndTime) {
        this.examEndTime = examEndTime;
    }

    public String getExamSuplement() {
        return examSuplement;
    }

    public void setExamSuplement(String examSuplement) {
        this.examSuplement = examSuplement;
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
        this.examOneTitle = examOneTitle;
    }

    public String getExamOneAnswer() {
        return examOneAnswer;
    }

    public void setExamOneAnswer(String examOneAnswer) {
        this.examOneAnswer = examOneAnswer;
    }

    public String getExamTwoTitle() {
        return examTwoTitle;
    }

    public void setExamTwoTitle(String examTwoTitle) {
        this.examTwoTitle = examTwoTitle;
    }

    public String getExamTwoAnswer() {
        return examTwoAnswer;
    }

    public void setExamTwoAnswer(String examTwoAnswer) {
        this.examTwoAnswer = examTwoAnswer;
    }

    public String getExamThreeTitle() {
        return examThreeTitle;
    }

    public void setExamThreeTitle(String examThreeTitle) {
        this.examThreeTitle = examThreeTitle;
    }

    public String getExamThreeAnswer() {
        return examThreeAnswer;
    }

    public void setExamThreeAnswer(String examThreeAnswer) {
        this.examThreeAnswer = examThreeAnswer;
    }

    public String getExamFourTitle() {
        return examFourTitle;
    }

    public void setExamFourTitle(String examFourTitle) {
        this.examFourTitle = examFourTitle;
    }

    public String getExamFourAnswer() {
        return examFourAnswer;
    }

    public void setExamFourAnswer(String examFourAnswer) {
        this.examFourAnswer = examFourAnswer;
    }

    public String getExamFiveTitle() {
        return examFiveTitle;
    }

    public void setExamFiveTitle(String examFiveTitle) {
        this.examFiveTitle = examFiveTitle;
    }

    public String getExamFiveAnswer() {
        return examFiveAnswer;
    }

    public void setExamFiveAnswer(String examFiveAnswer) {
        this.examFiveAnswer = examFiveAnswer;
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

    public String getOneSrc() {
        return oneSrc;
    }

    public void setOneSrc(String oneSrc) {
        this.oneSrc = oneSrc;
    }

    public String getTwoSrc() {
        return twoSrc;
    }

    public void setTwoSrc(String twoSrc) {
        this.twoSrc = twoSrc;
    }

    public String getThreeSrc() {
        return threeSrc;
    }

    public void setThreeSrc(String threeSrc) {
        this.threeSrc = threeSrc;
    }

    public String getFourSrc() {
        return fourSrc;
    }

    public void setFourSrc(String fourSrc) {
        this.fourSrc = fourSrc;
    }

    public String getFiveSrc() {
        return fiveSrc;
    }

    public void setFiveSrc(String fiveSrc) {
        this.fiveSrc = fiveSrc;
    }
    public void toString1(){
        System.out.println("StudentExam.toString():isEnd = [" + isEnd + "], examStartTime = [" + examStartTime + "], examEndTime = [" + examEndTime + "], examSuplement = [" + examSuplement + "], examLimitTime = [" + examLimitTime + "], examOneTitle = [" + examOneTitle + "], examOneAnswer = [" + examOneAnswer + "], examTwoTitle = [" + examTwoTitle + "], examTwoAnswer = [" + examTwoAnswer + "], examThreeTitle = [" + examThreeTitle + "], examThreeAnswer = [" + examThreeAnswer + "], examFourTitle = [" + examFourTitle + "], examFourAnswer = [" + examFourAnswer + "], examFiveTitle = [" + examFiveTitle + "], examFiveAnswer = [" + examFiveAnswer + "], examPublishTeacherId = [" + examPublishTeacherId + "], oneSrc = [" + oneSrc + "], twoSrc = [" + twoSrc + "], threeSrc = [" + threeSrc + "], fourSrc = [" + fourSrc + "], fiveSrc = [" + fiveSrc + "], examPass = [" + examPass + "]");
    }
}
