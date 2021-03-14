package com.xqq.pojo;

import java.io.Serializable;

public class Source implements Serializable {
    private Integer sourceId;

    private Integer belongUtilId;

    private String sourceSrc;

    private String sourceType;

    private String poster;

    private static final long serialVersionUID = 1L;

    public Integer getSourceId() {
        return sourceId;
    }

    public void setSourceId(Integer sourceId) {
        this.sourceId = sourceId;
    }

    public Integer getBelongUtilId() {
        return belongUtilId;
    }

    public void setBelongUtilId(Integer belongUtilId) {
        this.belongUtilId = belongUtilId;
    }

    public String getSourceSrc() {
        return sourceSrc;
    }

    public void setSourceSrc(String sourceSrc) {
        this.sourceSrc = sourceSrc == null ? null : sourceSrc.trim();
    }

    public String getSourceType() {
        return sourceType;
    }

    public void setSourceType(String sourceType) {
        this.sourceType = sourceType == null ? null : sourceType.trim();
    }

    public String getPoster() {
        return poster;
    }

    public void setPoster(String poster) {
        this.poster = poster == null ? null : poster.trim();
    }
}