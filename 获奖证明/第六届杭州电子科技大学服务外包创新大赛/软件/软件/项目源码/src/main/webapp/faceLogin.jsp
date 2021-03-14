<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/5/21 0021
  Time: 13:34
  To change this template use File | Settings | File Templates.
--%>
<%@ page language="java" import="java.util.*" pageEncoding="utf-8"%>
<%
    String path = request.getContextPath();
    String basePath = request.getScheme() + "://"
            + request.getServerName() + ":" + request.getServerPort()
            + path + "/";
%>

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
    <base href="<%=basePath%>">

    <title>管理员登录</title>
    <meta http-equiv="pragma" content="no-cache">
    <meta http-equiv="cache-control" content="no-cache">
    <meta http-equiv="expires" content="0">
    <meta http-equiv="keywords" content="keyword1,keyword2,keyword3">
    <meta http-equiv="description" content="This is my page">
    <script type="text/javascript" src="js/jquery-2.0.3.js"></script>
    <style>
        body {
            height: 100%;
            background: #213838;
            overflow: hidden;
        }

        canvas {
            z-index: -1;
            position: absolute;
        }
    </style>
    <script>
        $(document).ready(function() {
            //粒子背景特效
            $('body').particleground({
                dotColor: '#5cbdaa',
                lineColor: '#5cbdaa'
            });
        });
    </script>


    <style type="text/css">
        * {
            margin: 0;
            padding: 0;
        }

        body {
            height: 100vh;
            background-position: center;
            overflow: hidden;
        }

        h1 {
            color: #fff;
            text-align: center;
            font-weight: 100;
            margin-top: 40px;
        }

        #media {
            width: 50%;
            height: 50%;
            margin-top:10%;
		margin-left:25%;
		margin-right:24%;
            border-radius: 12%;
            overflow: hidden;
            opacity: 0.7;
        }

        #video {
		width:100%;

		height:80%;

        }

        #canvas {
           display:none;
        }
        #loginFaceBtn{
            background-color: #285e8e;
        }
        #loginFaceBtn:hover{
            cursor: pointer;
            background-color: #5bc0de;
        }

    </style>
</head>

<body>

<form method="post">
    <dl class="admin_login">
        <dt style="text-align: center">
            <strong>后台管理系统</strong><em>Management System</em> <strong>请把你的脸放摄像头面前</strong>
        </dt>
        <div id="media">
            <video id="video"  autoplay></video>
            <canvas id="canvas" width="400" height="300"></canvas>
        </div>
        <div style="text-align: center" >
            <input type="button" onclick="query()" value="立即登录"
                   style="width:15%;height:5%;margin-top: 1%;" id="loginFaceBtn" />
            <a href="faceRegist.jsp">人脸注册</a>
        </div>

    </dl>
    <script type="text/javascript">
        var video = document.getElementById("video"); //获取video标签
        var context = canvas.getContext("2d");
        var con  ={
            audio:false,
            video:{
                width:1980,
                height:1024,
            }
        };

        //导航 获取用户媒体对象 
       
           navigator.mediaDevices.getUserMedia(con).then(function(stream){
                video.srcObject = stream;
                video.onloadmetadate = function(e){
                    video.play();
                }
            });

	
        function query(){
            //把流媒体数据画到convas画布上去
            context.drawImage(video,0,0,400,300);
            var base = getBase64();
            $.ajax({
                type:"post",
                url:"faceLogin",
                data:{"baseData":base},
                success:function(data){
                    if(data=="manager"){
                        alert("登录成功")
                        window.location.href="manager.jsp";
                    }
                    else if(data=="fail"){
                        window.location.href="fail.jsp";
                    }
                    else {
                        alert("you have already login!")
                        window.location.href="manager.jsp";
                    }
                        ;
                    }
            });

        }
        function getBase64() {
            var imgSrc = document.getElementById("canvas").toDataURL(
                "image/png");
           // alert(imgSrc);
            return imgSrc.split("base64,")[1];
        };
    </script>
</form>
</body>
</html>

