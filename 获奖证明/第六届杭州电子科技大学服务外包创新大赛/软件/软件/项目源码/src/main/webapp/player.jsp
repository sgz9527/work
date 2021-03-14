<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/1/14 0014
  Time: 20:27
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>videoPlay</title>
</head>
<body>
<link href="http://vjs.zencdn.net/4.12/video-js.css" rel="stylesheet">
<script src="http://vjs.zencdn.net/4.12/video.js"></script>
<video id="MY_VIDEO_1" class="video-js vjs-default-skin" controls
       preload="auto" width="320" height="240" poster="img/t01a7616205d3c57213.png"
       data-setup="{}">
    <source src="video/mov_bbb.mp4" type='video/mp4'>
    <source src="video/mov_bbb.webm" type='video/webm'>
    <p class="vjs-no-js">To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="http://videojs.com/html5-video-support/" target="_blank">supports HTML5 video</a></p>
</video>
</body>
</html>
