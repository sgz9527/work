package com.xqq.pojo;

import org.springframework.stereotype.Component;

import java.io.Serializable;

/**
 * @author xqq
 */
@Component("SchOrCom")
public class SchOrCom implements Serializable {
    private Integer comId;

    private String comName;

    private String type;

    private String comBackground;

    private String comLogo;

    private String comPic;

    private String comIntroduce;

    private static final long serialVersionUID = 1L;

    public Integer getComId() {
        return comId;
    }

    public void setComId(Integer comId) {
        this.comId = comId;
    }

    public String getComName() {
        return comName;
    }

    public void setComName(String comName) {
        this.comName = comName == null ? null : comName.trim();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type == null ? null : type.trim();
    }

    public String getComBackground() {
        return comBackground;
    }

    public void setComBackground(String comBackground) {
        this.comBackground = comBackground == null ? null : comBackground.trim();
    }

    public String getComLogo() {
        return comLogo;
    }

    public void setComLogo(String comLogo) {
        this.comLogo = comLogo == null ? null : comLogo.trim();
    }

    public String getComPic() {
        return comPic;
    }

    public void setComPic(String comPic) {
        this.comPic = comPic == null ? null : comPic.trim();
    }

    public String getComIntroduce() {
        return comIntroduce;
    }

    public void setComIntroduce(String comIntroduce) {
        this.comIntroduce = comIntroduce == null ? null : comIntroduce.trim();
    }
}