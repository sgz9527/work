package com.xqq.pojo;

import java.io.Serializable;

public class Student implements Serializable {
    private Integer studentId;

    private String studentAccount;

    private String studentName;

    private String studentPassword;

    private String type;

    private String studentPic;

    private String studentIntroduce;

    private Integer commentNum;

    private String belongClass;

    private String belongSchId;

    private String createDate;

    private String studentEmail;

    private String belongSchName;

    private String studentSex;

    private Integer isBelongSch;

    private Integer breakTopicRuleNum;

    private Integer breakComRuleNum;

    private static final long serialVersionUID = 1L;

    public Integer getStudentId() {
        return studentId;
    }

    public void setStudentId(Integer studentId) {
        this.studentId = studentId;
    }

    public String getStudentAccount() {
        return studentAccount;
    }

    public void setStudentAccount(String studentAccount) {
        this.studentAccount = studentAccount == null ? null : studentAccount.trim();
    }

    public String getStudentName() {
        return studentName;
    }

    public void setStudentName(String studentName) {
        this.studentName = studentName == null ? null : studentName.trim();
    }

    public String getStudentPassword() {
        return studentPassword;
    }

    public void setStudentPassword(String studentPassword) {
        this.studentPassword = studentPassword == null ? null : studentPassword.trim();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type == null ? null : type.trim();
    }

    public String getStudentPic() {
        return studentPic;
    }

    public void setStudentPic(String studentPic) {
        this.studentPic = studentPic == null ? null : studentPic.trim();
    }

    public String getStudentIntroduce() {
        return studentIntroduce;
    }

    public void setStudentIntroduce(String studentIntroduce) {
        this.studentIntroduce = studentIntroduce == null ? null : studentIntroduce.trim();
    }

    public Integer getCommentNum() {
        return commentNum;
    }

    public void setCommentNum(Integer commentNum) {
        this.commentNum = commentNum;
    }

    public String getBelongClass() {
        return belongClass;
    }

    public void setBelongClass(String belongClass) {
        this.belongClass = belongClass == null ? null : belongClass.trim();
    }

    public String getBelongSchId() {
        return belongSchId;
    }

    public void setBelongSchId(String belongSchId) {
        this.belongSchId = belongSchId == null ? null : belongSchId.trim();
    }

    public String getCreateDate() {
        return createDate;
    }

    public void setCreateDate(String createDate) {
        this.createDate = createDate == null ? null : createDate.trim();
    }

    public String getStudentEmail() {
        return studentEmail;
    }

    public void setStudentEmail(String studentEmail) {
        this.studentEmail = studentEmail == null ? null : studentEmail.trim();
    }

    public String getBelongSchName() {
        return belongSchName;
    }

    public void setBelongSchName(String belongSchName) {
        this.belongSchName = belongSchName == null ? null : belongSchName.trim();
    }

    public String getStudentSex() {
        return studentSex;
    }

    public void setStudentSex(String studentSex) {
        this.studentSex = studentSex == null ? null : studentSex.trim();
    }

    public Integer getIsBelongSch() {
        return isBelongSch;
    }

    public void setIsBelongSch(Integer isBelongSch) {
        this.isBelongSch = isBelongSch;
    }

    public Integer getBreakTopicRuleNum() {
        return breakTopicRuleNum;
    }

    public void setBreakTopicRuleNum(Integer breakTopicRuleNum) {
        this.breakTopicRuleNum = breakTopicRuleNum;
    }

    public Integer getBreakComRuleNum() {
        return breakComRuleNum;
    }

    public void setBreakComRuleNum(Integer breakComRuleNum) {
        this.breakComRuleNum = breakComRuleNum;
    }
}