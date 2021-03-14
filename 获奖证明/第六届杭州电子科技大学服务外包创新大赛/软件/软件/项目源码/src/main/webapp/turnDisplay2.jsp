<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/1/18 0018
  Time: 21:46
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>

<head>
    <meta charset="utf-8" />
    <title>jquery版图片轮播</title>
    <link rel="stylesheet" href="css/test.css" />
</head>

<body>
<div class="container">
    <div class="inner clearfix">
        <div class="innerwraper"><img src="img/1.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/2.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/3.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/4.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/5.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/6.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/7.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/8.jpg" alt="" /></div>
        <div class="innerwraper"><img src="img/1.jpg" alt="" /></div>
    </div>
    <div class="pagination">
        <span class="active">1</span>
        <span>2</span>
        <span>3</span>
        <span>4</span>
        <span>5</span>
        <span>6</span>
        <span>7</span>
        <span>8</span>
    </div>
    <a href="javascript:void(0)" class="left-arrow">&lt;</a>
    <a href="javascript:void(0)" class="right-arrow">&gt;</a>
</div>
<script src="js/jquery-2.0.3.js"></script>
<script src="js/new.js"></script>
</body>
</html>
