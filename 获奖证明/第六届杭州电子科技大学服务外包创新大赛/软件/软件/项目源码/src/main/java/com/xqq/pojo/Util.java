package com.xqq.pojo;

import java.io.Serializable;

public class Util implements Serializable {
    private Integer utilId;

    private String utilTitle;

    private String utilName;

    private Integer belongChapterId;

    private Integer utilOrder;

    private static final long serialVersionUID = 1L;

    public Integer getUtilId() {
        return utilId;
    }

    public void setUtilId(Integer utilId) {
        this.utilId = utilId;
    }

    public String getUtilTitle() {
        return utilTitle;
    }

    public void setUtilTitle(String utilTitle) {
        this.utilTitle = utilTitle == null ? null : utilTitle.trim();
    }

    public String getUtilName() {
        return utilName;
    }

    public void setUtilName(String utilName) {
        this.utilName = utilName == null ? null : utilName.trim();
    }

    public Integer getBelongChapterId() {
        return belongChapterId;
    }

    public void setBelongChapterId(Integer belongChapterId) {
        this.belongChapterId = belongChapterId;
    }

    public Integer getUtilOrder() {
        return utilOrder;
    }

    public void setUtilOrder(Integer utilOrder) {
        this.utilOrder = utilOrder;
    }
}