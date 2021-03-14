package com.xqq.pojo;

import org.springframework.format.annotation.DateTimeFormat;

import java.io.Serializable;
import java.util.Date;

/**
 * @author xqq
 */
public class Test implements Serializable {
    private Integer testId;

    private String testName;

    private String testStartTime;
    @DateTimeFormat(pattern = "yyyy-MM-dd")
    private Date testEndTime;

    private Integer testAllMark;

    private Integer submitCount;

    private String testType;

    private String limitTime;

    private String testIntro;

    private Integer belongChapterId;

    private Integer publicTeacherId;

    private Short testState;
    private Chapter chapter;

    private static final long serialVersionUID = 1L;

    public Integer getTestId() {
        return testId;
    }

    public void setTestId(Integer testId) {
        this.testId = testId;
    }

    public String getTestName() {
        return testName;
    }

    public void setTestName(String testName) {
        this.testName = testName == null ? null : testName.trim();
    }

    public String getTestStartTime() {
        return testStartTime;
    }

    public void setTestStartTime(String testStartTime) {
        this.testStartTime = testStartTime == null ? null : testStartTime.trim();
    }

    public Date getTestEndTime() {
        return testEndTime;
    }

    public void setTestEndTime(Date testEndTime) {
        this.testEndTime = testEndTime;
    }

    public Integer getTestAllMark() {
        return testAllMark;
    }

    public void setTestAllMark(Integer testAllMark) {
        this.testAllMark = testAllMark;
    }

    public Integer getSubmitCount() {
        return submitCount;
    }

    public void setSubmitCount(Integer submitCount) {
        this.submitCount = submitCount;
    }

    public String getTestType() {
        return testType;
    }

    public void setTestType(String testType) {
        this.testType = testType == null ? null : testType.trim();
    }

    public String getLimitTime() {
        return limitTime;
    }

    public void setLimitTime(String limitTime) {
        this.limitTime = limitTime == null ? null : limitTime.trim();
    }

    public String getTestIntro() {
        return testIntro;
    }

    public void setTestIntro(String testIntro) {
        this.testIntro = testIntro == null ? null : testIntro.trim();
    }

    public Integer getBelongChapterId() {
        return belongChapterId;
    }

    public void setBelongChapterId(Integer belongChapterId) {
        this.belongChapterId = belongChapterId;
    }

    public Integer getPublicTeacherId() {
        return publicTeacherId;
    }

    public void setPublicTeacherId(Integer publicTeacherId) {
        this.publicTeacherId = publicTeacherId;
    }

    public Short getTestState() {
        return testState;
    }

    public void setTestState(Short testState) {
        this.testState = testState;
    }

    public Chapter getChapter() {
        return chapter;
    }

    public void setChapter(Chapter chapter) {
        this.chapter = chapter;
    }
}