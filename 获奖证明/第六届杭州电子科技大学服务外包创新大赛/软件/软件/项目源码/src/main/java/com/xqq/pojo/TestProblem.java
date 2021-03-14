package com.xqq.pojo;

import java.io.Serializable;

public class TestProblem implements Serializable {
    private Integer testProblemId;

    private Integer belongTestId;

    private Short testProblemType;

    private Integer testProblemOrder;

    private String testProblemTitle;

    private static final long serialVersionUID = 1L;

    public Integer getTestProblemId() {
        return testProblemId;
    }

    public void setTestProblemId(Integer testProblemId) {
        this.testProblemId = testProblemId;
    }

    public Integer getBelongTestId() {
        return belongTestId;
    }

    public void setBelongTestId(Integer belongTestId) {
        this.belongTestId = belongTestId;
    }

    public Short getTestProblemType() {
        return testProblemType;
    }

    public void setTestProblemType(Short testProblemType) {
        this.testProblemType = testProblemType;
    }

    public Integer getTestProblemOrder() {
        return testProblemOrder;
    }

    public void setTestProblemOrder(Integer testProblemOrder) {
        this.testProblemOrder = testProblemOrder;
    }

    public String getTestProblemTitle() {
        return testProblemTitle;
    }

    public void setTestProblemTitle(String testProblemTitle) {
        this.testProblemTitle = testProblemTitle == null ? null : testProblemTitle.trim();
    }
}