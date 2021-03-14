package com.xqq.pojo;

import java.io.Serializable;

public class smallProblem implements Serializable {
    private Integer smallProblemId;

    private Integer belongTestProblemId;

    private Short smallProblemType;

    private Short smallProblemScore;

    private String smallProblemADetail;

    private String smallProblemBDetail;

    private String smallProblemCDetail;

    private String smallProblemDDetail;

    private String smallProblemTip;

    private String smallProblemTrueAnswer;

    private String smallProblemTitle;

    private static final long serialVersionUID = 1L;

    public Integer getSmallProblemId() {
        return smallProblemId;
    }

    public void setSmallProblemId(Integer smallProblemId) {
        this.smallProblemId = smallProblemId;
    }

    public Integer getBelongTestProblemId() {
        return belongTestProblemId;
    }

    public void setBelongTestProblemId(Integer belongTestProblemId) {
        this.belongTestProblemId = belongTestProblemId;
    }

    public Short getSmallProblemType() {
        return smallProblemType;
    }

    public void setSmallProblemType(Short smallProblemType) {
        this.smallProblemType = smallProblemType;
    }

    public Short getSmallProblemScore() {
        return smallProblemScore;
    }

    public void setSmallProblemScore(Short smallProblemScore) {
        this.smallProblemScore = smallProblemScore;
    }

    public String getSmallProblemADetail() {
        return smallProblemADetail;
    }

    public void setSmallProblemADetail(String smallProblemADetail) {
        this.smallProblemADetail = smallProblemADetail == null ? null : smallProblemADetail.trim();
    }

    public String getSmallProblemBDetail() {
        return smallProblemBDetail;
    }

    public void setSmallProblemBDetail(String smallProblemBDetail) {
        this.smallProblemBDetail = smallProblemBDetail == null ? null : smallProblemBDetail.trim();
    }

    public String getSmallProblemCDetail() {
        return smallProblemCDetail;
    }

    public void setSmallProblemCDetail(String smallProblemCDetail) {
        this.smallProblemCDetail = smallProblemCDetail == null ? null : smallProblemCDetail.trim();
    }

    public String getSmallProblemDDetail() {
        return smallProblemDDetail;
    }

    public void setSmallProblemDDetail(String smallProblemDDetail) {
        this.smallProblemDDetail = smallProblemDDetail == null ? null : smallProblemDDetail.trim();
    }

    public String getSmallProblemTrueAnswer() {
        return smallProblemTrueAnswer;
    }

    public void setSmallProblemTrueAnswer(String smallProblemTrueAnswer) {
        this.smallProblemTrueAnswer = smallProblemTrueAnswer == null ? null : smallProblemTrueAnswer.trim();
    }

    public String getSmallProblemTitle() {
        return smallProblemTitle;
    }

    public void setSmallProblemTitle(String smallProblemTitle) {
        this.smallProblemTitle = smallProblemTitle == null ? null : smallProblemTitle.trim();
    }

    public String getSmallProblemTip() {
        return smallProblemTip;
    }

    public void setSmallProblemTip(String smallProblemTip) {
        this.smallProblemTip = smallProblemTip;
    }
}