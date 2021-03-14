<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/4/27 0027
  Time: 13:26
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.xqq.pojo.Teacher" %>
<%@ page import="com.xqq.pojo.Student" %>
<% Teacher teacher=(Teacher)session.getAttribute("teacherInfo");
    Student student=(Student)session.getAttribute("studentInfo");
    Integer studentId=-1;
    String  studentPic="";
    String studentName="";
    String studentPassword="";
    String studentSex="";
    String studentAccount="";
    String studentIntroduce="";
    String createDate="";
    String studentEmail="";
     if(student!=null)
    {
        studentId=student.getStudentId();
        studentName=student.getStudentName();
        studentPic=student.getStudentPic();
        studentPassword=student.getStudentPassword();
        studentAccount=student.getStudentAccount();
        studentIntroduce=student.getStudentIntroduce();
        studentSex=student.getStudentSex();
        createDate=student.getCreateDate();
        studentEmail=student.getStudentEmail();
    }
    //System.out.println("nullStudent");
%>
<html>
<head>
    <title>学生个人中心</title>
    <link rel="stylesheet" href="./css/bootstrap.min.css">
    <script src="js/jquery-2.0.3.js"></script>
    <script src="js/bootstrap.min.js"></script>
</head>
<style>
</style>
<body>
<form action="updateStudentInfo" method="post" id="submitStudentInfoForm"  style="margin-left:15%;margin-right:25%;margin-top:5%;" target="hidden_frame">
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">昵称:</div> <div style="width:78%;float:left;margin-left:1%;"><input type="text" name="studentName" value="<%=studentName%>" disabled id="studentName"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">账号:(不可修改)</div><div style="width:78%;float:left;margin-left:1%;"><input type="text"  value="<%=studentAccount%>" disabled id="studentAccount"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">创建日期:(不可修改)</div><div style="width:78%;float:left;margin-left:1%;"><input type="text"  value="<%=createDate%>" disabled id="createDate"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">密码:</div><div style="width:78%;float:left;margin-left:1%;"><input type="password" name="studentPassword" value="<%=studentPassword%>" disabled id="studentPassword"></div>
    </div>
    <div style="width:100%;height:50px;display:none;" id="checkDiv">
        <div style="width:15%;float:left;">再次确认密码:</div><div style="width:78%;float:left;margin-left:1%;"><input type="password"  id="studentPasswordCheck"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">邮箱:</div><div style="width:78%;float:left;margin-left:1%;"><input type="email" name="studentEmail" value="<%=studentEmail%>" disabled id="studentEmail"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">性别:</div><div style="width:78%;float:left;margin-left:1%;"><select name="studentSex" disabled id="studentSex"><option value="男">男</option><option value="女">女</option></select></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">头像:</div><div style="width:78%;float:left;margin-left:1%;height:auto;"><img  id="studentPic" src="img/<%=studentPic%>" style="width:7%;height:38px;"><span id="updateTexts" style="display:none;">点击修改头像</span><input type="file"  id="studentPicFile" style="display:none;"></div>
    </div>
    <div style="width:100%;height:100px;">
        <div style="width:15%;float:left;">简介:</div><div style="width:78%;float:left;margin-left:1%;"><textarea name="studentIntroduce" id="studentIntroduce" cols="50" rows="4" disabled><%=studentIntroduce%></textarea></div>
    </div>
    <input type="hidden" name="studentId" value="<%=studentId%>">
    <iframe name="hidden_frame" style="display:none;"></iframe>
    <input type="submit" style="display:none;" value="提交" id="submitBtn" onclick="return checkForm()">
</form>
<input type="button" value="点击修改" style="margin-left: 15%;" id="updateStudentInfoButton">
</body>
</html>
<script>
    var isUpdatePic=false;
    function checkForm(){
        if(isUpdatePic&&$("#studentPicFile").val()=="")
        {
            alert("未选择任何图片");
            return false;
        }
        if($("#studentPassword").val()!=$("#studentPasswordCheck").val())
        {
            alert("两次密码不一致");
            return false;
        }
        return true;
    }
    $(document).ready(function(){
        var isUpdate=false;
        $("#studentSex").val("<%=studentSex%>");
        $("#updateStudentInfoButton").click(function(){
            $("#studentEmail").attr("disabled",false);
            $("#updateTexts").show();
            isUpdate=true;
            $("#studentPassword").attr("disabled",false);
            $("#studentSex").attr("disabled",false);
            $("#studentName").attr("disabled",false);
            $("#updateStudentInfoButton").attr("disabled",true);
            $("#studentIntroduce").attr("disabled",false);
            $("#submitBtn").show();
            $("#studentPasswordCheck").val($("#studentPassword").val());
            $("#checkDiv").show();
        })
        $("#studentPic").click(function(){
            if(isUpdate){
                isUpdatePic=true;
                $("#studentPic").hide();
                $("#updateTexts").hide();
                $("#studentPicFile").show();
                $("#studentPicFile").attr("name","studentPicFile");
                $("#submitStudentInfoForm").attr("enctype","multipart/form-data");
            }
        })
    })
</script>
