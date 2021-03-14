<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/1/5 0005
  Time: 16:44
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>showComment</title>
    <script src="css/study.css"></script>
    <script src="js/jquery-2.0.3.js"></script>
    <script src="js/bootstrap.min.js"></script>
</head>
<body>
<form action="/showComment" method="post">
    topicId:<input type="text" name="topicId">
    <input type="submit" value="显示评论">
</form>
<div style="width:80% ;height:auto;background-color: azure;border: solid;border-width:1px;border-color: chartreuse" id="commentArea">
    <div id="comment" style="width:50%;">
        hello
    </div>
</div>
<form action="/writeComment" method="post">
    belongTopicId:<input type="text" name="belongTopicId">
    <br>
    <textarea name="commentInfo" id="commentInfo" cols="35" rows="5"></textarea>
    <br>
    writterId:<input type="text" name="commentWritterId" id="writterId">
    <br>
    <input type="hidden"  name="commentTime" id="commentTime">
    <br>
    学生：<input type="checkbox" name="commentWritterType" id="student" value="0" >
    老师：<input type="checkbox" name="commentWritterType" id="teacher" value="1">
    <br>
    <input type="submit" value="发表评论">
</form>
<form action="/praise" method="post">
   PraiseCommentId: <input type="text" name="commentId"><br>
    PraiseUserId:<input type="text" name="praiseUserId">
    <br>学生<input type="checkbox" name="praiseUserType" value="0">
    老师<input type="checkbox" name="praiseUserType" value="1"><br>
    <input type="submit" value="点赞">
</form>
<div class="adiscussrough">
    <div>
        <span>怎样做一个年少有为的当代青年？</span>
    </div>
    <div style="width:100%;height:auto;">
        <div style="float:left;width:40%;height:50px;">
            <span id="writer1" style="color:green;font-size:15px;">王纳莎</span>
            <span style="color:darkgrey;font-size:15px;margin-left:15px;">于2019-02-02 14:24:37发表</span>
            <span style="color:green;display:inline;font-size:14px;margin-left:15px;" id="topic1"class="topicDetail">详情</span>
        </div>
        <div style="float:left;width:59%;height:50px;">
            <span style="color:darkgrey;display:inline;font-size:14px;margin-left:470px;">浏览:34</span>
            <span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">评论:11</span>
            <span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">举报(0)</span>
        </div>
    </div>
</div>
</body>
<script>
    $(document).ready(function(){
        Date.prototype.Format = function (fmt) {
            var o = {
                "M+": this.getMonth() + 1, //月份
                "d+": this.getDate(), //日
                "H+": this.getHours(), //小时
                "m+": this.getMinutes(), //分
                "s+": this.getSeconds(), //秒
                "q+": Math.floor((this.getMonth() + 3) / 3), //季度
                "S": this.getMilliseconds() //毫秒
            };
            if (/(y+)/.test(fmt)) fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
            for (var k in o)
                if (new RegExp("(" + k + ")").test(fmt)) fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
            return fmt;
        }
        var time=new Date().Format("yyyy-MM-dd HH:mm:ss");
        $("#commentTime").val(time);
        alert($("#commentTime").val());
    })
</script>
</html>
