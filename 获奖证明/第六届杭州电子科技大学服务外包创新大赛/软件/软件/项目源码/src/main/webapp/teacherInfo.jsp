<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/4/27 0027
  Time: 18:45
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.xqq.pojo.Teacher" %>
<% Teacher teacher=(Teacher)session.getAttribute("teacherInfo");
    Integer teacherId=-1;
    String  teacherPic="";
    String teacherName="";
    String teacherPassword="";
    String teacherSex="";
    String teacherAccount="";
    String teacherIntroduce="";
    String createDate="";
    String teacherEmail="";
    if(teacher!=null)
    {
        teacherId=teacher.getTeacherId();
        teacherName=teacher.getTeacherName();
        teacherPic=teacher.getTeacherPic();
        teacherPassword=teacher.getTeacherPassword();
        teacherAccount=teacher.getTeacherAccount();
        teacherIntroduce=teacher.getTeacherIntroduce();
        teacherSex=teacher.getTeacherSex();
        createDate=teacher.getCreateDate();
        teacherEmail=teacher.getEmail();
    }
    //System.out.println("nullStudent");
%>
<html>
<head>
    <title>教师个人中心</title>
    <link rel="stylesheet" href="./css/bootstrap.min.css">
    <script src="js/jquery-2.0.3.js"></script>
    <script src="js/bootstrap.min.js"></script>
</head>
<body>
<form action="updateTeacherInfo" method="post" id="submitTeacherInfoForm"  style="margin-left:15%;margin-right:25%;margin-top:5%;" target="hidden_frame">
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">昵称:</div> <div style="width:78%;float:left;margin-left:1%;"><input type="text" name="teacherName" value="<%=teacherName%>" disabled id="teacherName"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">账号:(不可修改)</div><div style="width:78%;float:left;margin-left:1%;"><input type="text"  value="<%=teacherAccount%>" disabled id="teacherAccount"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">创建日期:(不可修改)</div><div style="width:78%;float:left;margin-left:1%;"><input type="text"  value="<%=createDate%>" disabled id="createDate"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">密码:</div><div style="width:78%;float:left;margin-left:1%;"><input type="password" name="teacherPassword" value="<%=teacherPassword%>" disabled id="teacherPassword"></div>
    </div>
    <div style="width:100%;height:50px;display:none;" id="checkDiv">
        <div style="width:15%;float:left;">再次确认密码:</div><div style="width:78%;float:left;margin-left:1%;"><input type="password"  id="teacherPasswordCheck"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">邮箱:</div><div style="width:78%;float:left;margin-left:1%;"><input type="email" name="email" value="<%=teacherEmail%>" disabled id="teacherEmail"></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">性别:</div><div style="width:78%;float:left;margin-left:1%;"><select name="teacherSex" disabled id="teacherSex"><option value="男">男</option><option value="女">女</option></select></div>
    </div>
    <div style="width:100%;height:50px;">
        <div style="width:15%;float:left;">头像:</div><div style="width:78%;float:left;margin-left:1%;height:auto;"><img  id="teacherPic" src="img/<%=teacherPic%>" style="width:7%;height:38px;"><span id="updateTexts" style="display:none;">点击修改头像</span><input type="file"  id="teacherPicFile" style="display:none;"></div>
    </div>
    <div style="width:100%;height:100px;">
        <div style="width:15%;float:left;">简介:</div><div style="width:78%;float:left;margin-left:1%;"><textarea name="teacherIntroduce" id="teacherIntroduce" cols="50" rows="4" disabled><%=teacherIntroduce%></textarea></div>
    </div>
    <input type="hidden" name="teacherId" value="<%=teacherId%>">
    <iframe name="hidden_frame" style="display:none;"></iframe>
    <input type="submit" style="display:none;" value="提交" id="submitBtn" onclick="return checkForm()">
</form>
<input type="button" value="点击修改" style="margin-left: 15%;" id="updateTeacherInfoButton">
</body>
</html>
<script>
    var isUpdatePic=false;
    function checkForm(){
        if(isUpdatePic&&$("#teacherPicFile").val()=="")
        {
            alert("未选择任何图片");
            return false;
        }
        if($("#teacherPassword").val()!=$("#teacherPasswordCheck").val())
        {
            alert("两次密码不一致");
            return false;
        }
        return true;
    }
    $(document).ready(function(){
        var isUpdate=false;
        $("#teacherSex").val("<%=teacherSex%>");
        $("#updateTeacherInfoButton").click(function(){
            $("#teacherEmail").attr("disabled",false);
            $("#updateTexts").show();
            isUpdate=true;
            $("#teacherPassword").attr("disabled",false);
            $("#teacherSex").attr("disabled",false);
            $("#teacherName").attr("disabled",false);
            $("#updateTeacherInfoButton").attr("disabled",true);
            $("#teacherIntroduce").attr("disabled",false);
            $("#submitBtn").show();
            $("#teacherPasswordCheck").val($("#teacherPassword").val());
            $("#checkDiv").show();
        })
        $("#teacherPic").click(function(){
            if(isUpdate){
                isUpdatePic=true;
                $("#teacherPic").hide();
                $("#updateTexts").hide();
                $("#teacherPicFile").show();
                $("#teacherPicFile").attr("name","teacherPicFile");
                $("#submitTeacherInfoForm").attr("enctype","multipart/form-data");
            }
        })
    })
</script>
