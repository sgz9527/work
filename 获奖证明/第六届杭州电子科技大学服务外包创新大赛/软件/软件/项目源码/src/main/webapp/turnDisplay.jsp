<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/1/18 0018
  Time: 21:20
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>

<head>
    <title>jQuery slideViewerPro 1.0 Demo by http://www.codefans.net</title>
    <meta http-equiv="Content-Type" content="text/html; charset=gb2312" />
    <link rel="stylesheet" href="css/svwp_style.css" type="text/css" media="screen" />
    <script src="js/jquery-2.0.3.js" type="text/javascript"></script>
    <script src="js/Jquery%20slideViewerPro%201.5.js" type="text/javascript"></script>
    <script type="text/javascript">
        $(window).bind("load", function() {
            $("div#my-folio-of-works").slideViewerPro({
                thumbs: 4,
                autoslide: true,  //自动切换
                asTimer: 3500,   //切换时间
                typo: true,
                galBorderWidth: 0, //大图边框
                thumbsBorderOpacity: 0,
                buttonsTextColor: "#707070",
                buttonsWidth: 10,
                thumbsRightMargin: 5,//缩略图之间的间隙
                thumbsPercentReduction: 22,//缩略图是原图的22%，与buttonsWidth对应，越少，%分比则可增加
                thumbsActiveBorderOpacity: 0.8,//透明度
                thumbsActiveBorderColor: "aqua",
                thumbsActiveBorderColor: "#ff0000",
                shuffle: false  //随机排序
            }); });
    </script>
    <script src="js/jquery.timers.js" type="text/javascript"></script>
</head>
<body>
<div id="my-folio-of-works" class="svwp">
    <ul>
        <li><img alt="会议1"  src="./img/4.jpg" width="400" height="300" /></li>
        <li><img alt="会议2"  src="img/5.jpg" width="400" height="300"  /></li>
        <li><img alt="会议3"  src="img/6.jpg" width="400" height="300"  /></li>
        <li><img alt="会议4"  src="img/7.jpg" width="400" height="300"  /></li>
        <!--eccetera-->
    </ul>
</div>
</body>
</html>
