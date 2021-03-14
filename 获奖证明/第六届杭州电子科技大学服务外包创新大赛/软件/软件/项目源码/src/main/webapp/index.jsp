<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.xqq.pojo.Teacher" %>
<%@ page import="com.xqq.pojo.Student" %>
<% Teacher teacher=(Teacher)session.getAttribute("teacherInfo");
    Student student=(Student)session.getAttribute("studentInfo");
    Integer studentId=-1;
    String  studentPic="";
    String studentName="";
    String teacherName="";
    String teacherPic="";
    Integer teacherId=-1;
    if(teacher!=null )
    {
        teacherId=teacher.getTeacherId();
        teacherName=teacher.getTeacherName();
        teacherPic=teacher.getTeacherPic();
    }
    else if(student!=null)
    {
        studentId=student.getStudentId();
        studentName=student.getStudentName();
        studentPic=student.getStudentPic();
    }
%>
<html>
<head >
    <link href="css/bootstrap.min.css"  rel="stylesheet" type="text/css">
    <script src="js/jquery-2.0.3.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <link href="css/index.css" rel="stylesheet" type="text/css">
    <title>云智教育平台首页</title>
    <style type="text/css">
        *{
            margin:0 auto;
            padding:0;
        }
        ul{
            list-style-type:none;
        }
        .branch_bar{
            width:750px;
            height:220px;
            overflow:hidden;
            position:relative;
        }
        .branch_bar .branch{
            width:3000px;
        }
        .branch_bar .branch li{
            float:left;
        }
        .branch_bar .branch li .lii{
            width:750px;
            height:220px;
            border:0;
        }
    </style>
</head>
<body>
<div class="headall">
    <div class="head" >
        <ul >
            <li style="margin-left:1px;"><img src="img/151946214719325.png" style="height:32px;position: relative;top:0px;background: rgba(229,219,219,1.00);"></li>
            <li >
                <a href="index.jsp" >首页</a>
            </li>
            <li >
                <a href="information.jsp" >资讯</a>
            </li>
            <li >
                <a href="excercise.jsp" >实训</a>
            </li>
            <li >
                <a href="companyintroduce.jsp" >企业</a>
            </li>
            <li >
                <a href="contact.jsp" >交流</a>
            </li>
            <li><a href="adminLogin.jsp">网站管理</a></li>
        </ul>
    </div>
    <div class="rightofhead">
        <div style="width:80%;float: left;height:inherit;">
            <input class="text" id="searchText" value="在这里搜索课程" onfocus="if (value =='在这里搜索课程'){value =''}" onblur="if (value ==''){value='在这里搜索课程'}" />
            <input type="button" class="searchbutton" value="搜索" id="button1" >
        </div>
        <div style="width:20%;float:left;height:inherit;padding-top:8px;padding-right:0px;" id="beforeLogin">
            <a  data-target="#signin" data-toggle="modal" >登录</a>
            <a  data-target="#signup" data-toggle="modal">注册</a>
        </div>
        <div style="width:20%;float:left;height:inherit;padding-top:4px;padding-right:0px;display:none;" id="afterLogin" data-toggle="popover" title="???" data-placement="bottom" >
            <img src="." style="height:30px;width:30px;border-radius:15px;border-style:none;" id="Pic"/>
        </div>

    </div>
</div>
<!--head部分结束-->
<div id="between" style="width:98%;height:16%;margin-top:3%;background-color:whitesmoke;margin-left:1%;margin-left:1%;">
    <div style="width:50%;height:90px;float:left">
    <img src="img/151946214719325.png" style="width:60%;height:inherit;margin-left:5%;">
    </div>
    <div style="width:50%;float:left;heigth:90px;">
        <h2 style="color:#7E9E9C ;font-style: italic;">我们不生产知识</h2>
        <h2 style="color:#7E9E9C ;font-style: italic;">我们只是知识的搬运工</h2>
    </div>
</div>
<div id="myCarousel1" class="carousel slide" >
    <!-- 轮播（Carousel）指标 -->
    <ol class="carousel-indicators">
        <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
        <li data-target="#myCarousel" data-slide-to="1"></li>
        <li data-target="#myCarousel" data-slide-to="2"></li>
    </ol>
    <!-- 轮播（Carousel）项目 -->
    <div class="carousel-inner" style="height:400px;width:100%;margin-top:1%;position:absolute;display:none;" id="turnDisPlay">
        <div class="item active">
            <img src="img/201901090946311000.png" alt="First slide" style="width:100%;height:inherit">
        </div>
        <div class="item">
            <img src="img/201812211823071000.jpg" alt="Second slide" style="width:100%;height:inherit">
        </div>
        <div class="item">
            <img src="img/201812211822101000.jpg" alt="Third slide" style="width:100%;height:inherit">
        </div>
    </div>
    <!-- 轮播（Carousel）导航 -->
    <a class="left carousel-control" href="#myCarousel1" role="button" data-slide="prev">
        <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
        <span class="sr-only">&rsaquo;</span>
    </a>
    <a class="right carousel-control" href="#myCarousel1" role="button" data-slide="next">
        <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
        <span class="sr-only" >&rsaquo;</span>
    </a>
    <div class="rightbmg">
        <div style="background:aliceblue;">
            <img src="img/loginByFace.png" id="logocomorsch" style="height:55px;margin-left:10%;margin-right:10%;">
            <p style="display:none;position:fixed;background-color:snow;opacity: 1.0;color:dimgray;font-size: 15px;width:32%;">这里是刷脸登录入口，你可以选择通过摄取人脸的方式来进行登录</p>
        </div>

        <div>
            <img src="img/faceRecon.jpg"  style="height:120px;width:100%;">
        </div>
        <div>
        <div ><input class="login" type="button" id="loginbutton" value="立即刷脸登录"></div>
    </div>
</div>
</div>
<div class="block" style="position:relative;top:160px;">
    <div  style="margin-top:42px;text-align:center;width:100%;height:80px;"><img src="./img/151946344658118.png"></div>
    <div class="content">
        <div >
            <img src="./img/introduceContainer1.gif" style="height: 200px;">
        </div>
        <div style="text-align: center"><p style="font-size: 18px;">海量课程资源</p></div>
    </div>
    <div class="content">
        <div >
            <img src="img/introduceContainer2.gif" style="height: 200px;">
        </div>
        <div style="text-align: center"><p style="font-size: 18px;">观看视频无卡顿</p></div>
    </div>
    <div class="content">
        <div >
            <img src="img/introduceContainer3.gif" style="height: 200px;">
        </div>
        <div style="text-align: center"><p style="font-size: 18px;">每天都有名师学术探讨</p></div>
    </div>
    <div class="content">
        <div >
            <img src="img/introduceContainer4.gif" style="height: 200px;">
        </div>
        <div style="text-align: center"><p style="font-size: 18px;">点滴记录真实学习动态</p></div>
    </div>
</div>
<!--平台优势展示结束-->
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
<div style="width:100%;height:200px;display:none;position:relative;top:130px;" id="recommandCourses">
    <h3 style="color:black;margin-left:3%;">推荐课程</h3>
    <div style="width:98%;margin-left:2%;height:160px;" id="recomCourse">
        <div class="recommendCourse" style="float:left;margin-left:1%;width:100%;height:auto;">
            <div class="content" style="margin-left:0px;display:none" id="content1">
                <div class="course-item-inner">
                    <a data-most="0" href="" target="_blank" id="coursehref13" >
                        <div class="coursebgm" id="coursebgm13" >
                            <span class="course-status" id="course-status13"></span>
                        </div>
                    </a>
                </div>
                <div class="course-footer" id="coursefooter13">
                    <a href="" style="text-decoration: none" id="course-footerhref13"><h3 class="course-title" id="course-title13"></h3></a>
                    <span class="course-author" id="course-author13"><strong></strong></span>
                    <span class="course-fans" id="fans13"> </span>
                    <span class="course-comments" id="comments13"></span>
                    <div class="course-intro" id="course-intro13"></div>
                </div>
            </div>
            <div class="content" style="margin-left:33px;display:none;" id="content2">
                <div class="course-item-inner">
                    <a data-most="0" href="" target="_blank" id="coursehref14" >
                        <div class="coursebgm" id="coursebgm14" >
                            <span class="course-status" id="course-status14"></span>
                        </div>
                    </a>
                </div>
                <div class="course-footer" id="coursefooter14">
                    <a href="" style="text-decoration: none" id="course-footerhref14"><h3 class="course-title" id="course-title14"></h3></a>
                    <span class="course-author" id="course-author14"><strong></strong></span>
                    <span class="course-fans" id="fans14"> </span>
                    <span class="course-comments" id="comments14"></span>
                    <div class="course-intro" id="course-intro14"></div>
                </div>
            </div>
            <div class="content" style="margin-left:33px;display:none;" id="content3">
                <div class="course-item-inner">
                    <a data-most="0" href="" target="_blank" id="coursehref15" >
                        <div class="coursebgm" id="coursebgm15" >
                            <span class="course-status" id="course-status15"></span>
                        </div>
                    </a>
                </div>
                <div class="course-footer" id="coursefooter15">
                    <a href="" style="text-decoration: none" id="course-footerhref15"><h3 class="course-title" id="course-title15"></h3></a>
                    <span class="course-author" id="course-author15"><strong></strong></span>
                    <span class="course-fans" id="fans15"> </span>
                    <span class="course-comments" id="comments15"></span>
                    <div class="course-intro" id="course-intro15"></div>
                </div>
            </div>
            <div class="content" style="margin-left:33px;display:none;" id="content4">
                <div class="course-item-inner">
                    <a data-most="0" href="" target="_blank" id="coursehref16" >
                        <div class="coursebgm" id="coursebgm16" >
                            <span class="course-status" id="course-status16"></span>
                        </div>
                    </a>
                </div>
                <div class="course-footer" id="coursefooter16">
                    <a href="" style="text-decoration: none" id="course-footerhref16"><h3 class="course-title" id="course-title16"></h3></a>
                    <span class="course-author" id="course-author16"><strong></strong></span>
                    <span class="course-fans" id="fans16"> </span>
                    <span class="course-comments" id="comments16"></span>
                    <div class="course-intro" id="course-intro16"></div>
                </div>
            </div>
        </div>

    </div>
</div>
<div class="title" style="margin-top:270px;"><img src="img/151947018388475.png"></div>
<div class="block">
    <div class="blockraw">
        <div class="content" style="margin-left:33px;">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref1" >
                    <div class="coursebgm" id="coursebgm1" >
                        <span class="course-status" id="course-status1"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter1">
                <a href="" style="text-decoration: none" id="course-footerhref1"><h3 class="course-title" id="course-title1"></h3></a>
                <span class="course-author" id="course-author1"><strong></strong></span>
                <span class="course-fans" id="fans1"> </span>
                <span class="course-comments" id="comments1"></span>
                <div class="course-intro" id="course-intro1"></div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref2" >
                    <div class="coursebgm" id="coursebgm2" >
                        <span class="course-status" id="course-status2"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter2">
                <a href="" style="text-decoration: none" id="course-footerhref2"><h3 class="course-title" id="course-title2"></h3></a>
                <span class="course-author" id="course-author2"><strong></strong></span>
                <span class="course-fans" id="fans2"> </span>
                <span class="course-comments" id="comments2"></span>
                <div class="course-intro" id="course-intro2">
                </div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref3" >
                    <div class="coursebgm" id="coursebgm3" >
                        <span class="course-status" id="course-status3"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter3">
                <a href="" style="text-decoration: none" id="course-footerhref3"><h3 class="course-title" id="course-title3"></h3></a>
                <span class="course-author" id="course-author3"><strong></strong></span>
                <span class="course-fans" id="fans3"> </span>
                <span class="course-comments" id="comments3"></span>
                <div class="course-intro" id="course-intro3"></div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref4">
                    <div class="coursebgm" id="coursebgm4" >
                        <span class="course-status" id="course-status4"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter4">
                <a href="" style="text-decoration: none" id="course-footerhref4"><h3 class="course-title" id="course-title4"></h3></a>
                <span class="course-author" id="course-author4"><strong></strong></span>
                <span class="course-fans" id="fans4"> </span>
                <span class="course-comments" id="comments4"></span>
                <div class="course-intro" id="course-intro4">
                </div>
            </div>
        </div>
    </div>
    <div class="blockraw">
        <div class="content" style="margin-left:33px;">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref5" >
                    <div class="coursebgm" id="coursebgm5" >
                        <span class="course-status" id="course-status5"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter5">
                <a href="" style="text-decoration: none" id="course-footerhref5"><h3
                        class="course-title" id="course-title5"></h3></a>
                <span class="course-author" id="course-author5"><strong></strong></span>
                <span class="course-fans" id="fans5"> </span>
                <span class="course-comments" id="comments5"></span>
                <div class="course-intro" id="course-intro5">
                </div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref6">
                    <div class="coursebgm" id="coursebgm6" >
                        <span class="course-status" id="course-status6"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter6">
                <a href="" style="text-decoration: none" id="course-footerhref6"><h3 class="course-title" id="course-title6"></h3></a>
                <span class="course-author" id="course-author6"><strong></strong></span>
                <span class="course-fans" id="fans6"> </span>
                <span class="course-comments" id="comments6"></span>
                <div class="course-intro" id="course-intro6">
                </div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref7" >
                    <div class="coursebgm" id="coursebgm7" >
                        <span class="course-status" id="course-status7"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter7">
                <a href="" style="text-decoration: none" id="course-footerhref7"><h3 class="course-title" id="course-title7"></h3></a>
                <span class="course-author" id="course-author7"><strong></strong></span>
                <span class="course-fans" id="fans7"> </span>
                <span class="course-comments" id="comments7"></span>
                <div class="course-intro" id="course-intro7">
                </div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref8" >
                    <div class="coursebgm" id="coursebgm8" >
                        <span class="course-status" id="course-status8"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter8">
                <a href="" style="text-decoration: none" id="course-footerhref8"><h3 class="course-title" id="course-title8"></h3></a>
                <span class="course-author" id="course-author8"><strong></strong></span>
                <span class="course-fans" id="fans8"> </span>
                <span class="course-comments" id="comments8"></span>
                <div class="course-intro" id="course-intro8"></div>
            </div>
        </div>
    </div>
    <div class="blockraw">
        <div class="content" style="margin-left:33px;">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref9">
                    <div class="coursebgm" id="coursebgm9"  >
                        <span class="course-status" id="course-status9"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter9">
                <a href="" style="text-decoration: none" id="course-footerhref9"><h3 class="course-title" id="course-title9"></h3></a>
                <span class="course-author" id="course-author9"><strong></strong></span>
                <span class="course-fans" id="fans9"> </span>
                <span class="course-comments" id="comments9"></span>
                <div class="course-intro" id="course-intro9">
                </div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref10">
                    <div class="coursebgm" id="coursebgm10" >
                        <span class="course-status" id="course-status10"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter10">
                <a href="" style="text-decoration: none" id="course-footerhref10"><h3 class="course-title" id="course-title10"></h3></a>
                <span class="course-author" id="course-author10"><strong></strong></span>
                <span class="course-fans" id="fans10"> </span>
                <span class="course-comments" id="comments10"></span>
                <div class="course-intro" id="course-intro10"></div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref11">
                    <div class="coursebgm" id="coursebgm11" >
                        <span class="course-status" id="course-status11"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter11">
                <a href="" style="text-decoration: none" id="course-footerhref11"><h3 class="course-title" id="course-title11"></h3></a>
                <span class="course-author" id="course-author11"><strong></strong></span>
                <span class="course-fans" id="fans11"> </span>
                <span class="course-comments" id="comments11"></span>
                <div class="course-intro" id="course-intro11">
                </div>
            </div>
        </div>
        <div class="content">
            <div class="course-item-inner">
                <a data-most="0" href="" target="_blank" id="coursehref12">
                    <div class="coursebgm" id="coursebgm12" >
                        <span class="course-status" id="course-status12"></span>
                    </div>
                </a>
            </div>
            <div class="course-footer" id="coursefooter12">
                <a href="" style="text-decoration: none" id="course-footerhref12"><h3 class="course-title" id="course-title12"></h3></a>
                <span class="course-author" id="course-author12"><strong></strong></span>
                <span class="course-fans" id="fans12"> </span>
                <span class="course-comments" id="comments12"></span>
                <div class="course-intro" id="course-intro12">
                </div>
            </div>
        </div>


    </div>
</div>
<div style="width:100%;height:auto;margin-bottom:20px;margin-top:50px;">
    <a href="category.jsp" style="margin-left:33%;"><img src="img/courseall.png" class="courseall" ></a>
</div>
<!--课程资源展示结束-->





<div class="title"><img src="img/taolun.png" ></div>
<div class="discuss">
    <div class="leftdiscuss">
        <div class="logoofdiscuss"></div>
        <div style="text-align: center"><h3 style="color:white;font-size:20px;">一起学习 一起成长</h3></div>
        <div class="join"><a style="text-decoration: none;color:rgba(235,235,235,0.86);">加入我们</a></div>
    </div>
    <div class="carousel slide" id="myCarousel" style="width:65%;height:195px;float:left;">

        <ol class="carousel-indicators" >
            <li data-target="#myCarousel" data-slide-to="0" class="active" style="background-color:orange;"></li>
            <li data-target="#myCarousel" data-slide-to="1" style="background-color:orange;"></li>
            <li data-target="#myCarousel" data-slide-to="2" style="background-color:orange;"></li>
        </ol>
        <div class="carousel-inner">
            <div class="item active">
                <div class="discuss1">
                    <div class="headofdiscuss">
                        <div class="lefthead">
                            <span></span>
                            <span>Alice</span>
                            <span>老师</span>
                        </div>
                        <div class="righthead">来自    <span style="color:dimgrey">计算机组成原理</span></div>
                    </div>
                    <div class="discussdetails">计算机组成原理怎么学？</div>
                    <div class="discussfooter">已有<span style="color:peru">2347</span>人参加该讨论话题</div>
                </div>
                <div class="discuss2">
                    <div class="headofdiscuss">
                        <div class="lefthead">
                            <span></span>
                            <span>张三</span>
                            <span>学员</span>
                        </div>
                        <div class="righthead">来自    <span style="color:dimgrey">C语言程序设计</span></div>
                    </div>
                    <div class="discussdetails">怎么学习C语言?</div>
                    <div class="discussfooter">已有<span style="color:peru">2344</span>人参加该讨论话题</div>
                </div>
            </div>



            <div class="item">
                <div class="discuss1">
                    <div class="headofdiscuss">
                        <div class="lefthead">
                            <span></span>
                            <span>Alice</span>
                            <span>老师</span>
                        </div>
                        <div class="righthead">来自    <span style="color:dimgrey">我的电竞人生</span></div>
                    </div>
                    <div class="discussdetails">怎么打线上运营</div>
                    <div class="discussfooter">已有<span style="color:peru">2347</span>人参加该讨论话题</div>
                </div>
                <div class="discuss2">
                    <div class="headofdiscuss">
                        <div class="lefthead">
                            <span></span>
                            <span>张三</span>
                            <span>学员</span>
                        </div>
                        <div class="righthead">来自    <span style="color:dimgrey">数据结构</span></div>
                    </div>
                    <div class="discussdetails">最短路径C语言实现？</div>
                    <div class="discussfooter">已有<span style="color:peru">2344</span>人参加该讨论话题</div>
                </div>
            </div>



            <div class="item " >
                <div class="discuss1">
                    <div class="headofdiscuss">
                        <div class="lefthead">
                            <span></span>
                            <span>Alice</span>
                            <span>老师</span>
                        </div>
                        <div class="righthead">来自    <span style="color:dimgrey">数据结构</span></div>
                    </div>
                    <div class="discussdetails">怎么求二叉树的宽度？</div>
                    <div class="discussfooter">已有<span style="color:peru">2347</span>人参加该讨论话题</div>
                </div>
                <div class="discuss2">
                    <div class="headofdiscuss">
                        <div class="lefthead">
                            <span></span>
                            <span>张三</span>
                            <span>学员</span>
                        </div>
                        <div class="righthead">来自    <span style="color:dimgrey">探索土星上的生命</span></div>
                    </div>
                    <div class="discussdetails">这门课的意义是什么？</div>
                    <div class="discussfooter">已有<span style="color:peru">2344</span>人参加该讨论话题</div>
                </div>
            </div>
        </div>
        <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">
            <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
            <span class="sr-only">Previous</span>
        </a>
        <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">
            <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
            <span class="sr-only">Next</span>
        </a>
    </div>
</div>
<!-- 课程交流区结束-->
<div class="title"><img src="img/152221718443679.png" ></div>
<div style="width:100%;height:320px;">
    <div style="width:inherit;height:190px;">
        <div style="float:left;width:12%;height:auto;margin-left:50px;">
            <a href="" id="comhref1">
                <img style="width:100%;height:132px;border-radius: 75px;text-align:center;" src="." id="comimg1">
                <span style="margin-top:30px;color:black;font-size:16px;text-align:center;margin-left:26px;" id="comname1"></span>
            </a>
        </div>
        <div style="float:left;width:12%;height:auto;margin-left:50px;">
            <a href="" id="comhref2">
                <img style="width:100%;height:132px;border-radius: 75px;text-align:center;" src="." id="comimg2">
                <span style="margin-top:30px;color:black;font-size:16px;text-align:center;margin-left:26px;" id="comname2"></span>
            </a>
        </div>
        <div style="float:left;width:12%;height:auto;margin-left:50px;">
            <a href="" id="comhref3">
                <img style="width:100%;height:132px;border-radius: 75px;text-align:center;" src="." id="comimg3">
                <span style="margin-top:30px;color:black;font-size:16px;text-align:center;margin-left:26px;" id="comname3"></span>
            </a>
        </div>
        <div style="float:left;width:12%;height:auto;margin-left:50px;">
            <a href="" id="comhref4">
                <img style="width:100%;height:132px;border-radius: 75px;text-align:center;" src="." id="comimg4">
                <span style="margin-top:30px;color:black;font-size:16px;text-align:center;margin-left:26px;" id="comname4"></span>
            </a>
        </div>
        <div style="float:left;width:12%;height:auto;margin-left:50px;">
            <a href="" id="comhref5">
                <img style="width:100%;height:132px;border-radius: 75px;text-align:center;" src="." id="comimg5">
                <span style="margin-top:30px;color:black;font-size:16px;text-align:center;margin-left:26px;" id="comname5"></span>
            </a>
        </div>
        <div style="float:left;width:12%;height:auto;margin-left:50px;">
            <a href="" id="comhref6">
                <img style="width:100%;height:142px;border-radius:75px;text-align:center;" src="." id="comimg6">
                <span style="margin-top:30px;color:black;font-size:16px;text-align:center;margin-left:26px;weidth:100%;" id="comname6"></span>
            </a>
        </div>
    </div>
    <div style="width:100%;height:90px;">
        <input type="button" style="background:green;color:white;border-style:none;width:30%;height:70px;margin-left:35%;margin-right:35%;margin-top:25px;font-size:20px;" value="查看所有合作院校机构" id="viewallsch">
    </div>
</div>

<!--用户学习区结束-->

<div class="footer">
    <div class="container">
        <div class="row">
            <div class="col-md-4 clearfix footer-col">
                <img src="img/151946214719325.png" style="height:50px;">
                <div class="footer-slogan" style="color:white;">竭力打造最优质的平台
                </div>
                <div class="col-xs-2">
                    <div class="social-item footer-weixin-item">
                        <i class="fa fa-weixin"></i>
                        <div class="footer-weixin">
                            <img src="img/weixin1.jpg" style="height:40px;">
                        </div>
                    </div>
                </div>
                <div class="col-xs-2">
                    <div class="social-item footer-qq-item">
                        <i class="fa fa-qq"></i>
                        <div class="footer-weixin">
                            <img src="img/qq11.png" style="height:40px;">
                        </div>
                    </div>
                </div>
                <div class="col-xs-2">
                    <div class="social-item footer-weibo-item">
                        <a href="http://weibo.com/shiyanlou2013" target="_blank">
                            <i class="fa fa-weibo"></i>
                            <div class="footer-weixin">
                                <img src="img/weibo1.jpg" style="height:40px;">
                            </div>
                        </a>
                    </div>
                </div>
                <div class="col-xs-2">
                    <div class="social-item footer-weibo-item">
                        <a href="http://weibo.com/shiyanlou2013" target="_blank">
                            <i class="fa fa-weibo"></i>
                            <div class="footer-weixin">
                                <img src="img/qqq.png" style="height:40px;">
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            <div class="col-xs-6 col-sm-3 col-md-2 footer-col" style="color:grey;">
                <div class="col-title" style="color:white;">公司</div>
                <a href="privacy/index.html" target="_blank">关于我们</a><br>
                <a href="privacy/index.html" target="_blank">联系我们</a><br>
                <a href="http://www.simplecloud.cn/jobs.html" target="_blank">加入我们</a><br>
                <a href="https://blog.shiyanlou.com" target="_blank">技术博客</a><br>
            </div>
            <div class="col-xs-6 col-sm-3 col-md-2 footer-col" style="color:grey;">
                <div class="col-title" style="color:white;">合作</div>
                <a href="privacy/index.html" target="_blank">我要投稿</a><br>
                <a href="labs/index.html" target="_blank">教师合作</a><br>
                <a href="edu/index.html" target="_blank">高校合作</a><br>
                <a href="privacy/index.html" target="_blank">友情链接</a>
            </div>
            <div class="col-xs-6 col-sm-3 col-md-2 footer-col" style="color:grey;">
                <div class="col-title" style="color:white;">服务</div>
                <a href="bootcamp/index.html" target="_blank">综合实训</a><br>
                <a href="#a" target="_blank">了解课程</a><br>
                <a href="questions/index.html?tag=%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98" target="_blank">常见问题</a><br>
                <a href="privacy/index.html" target="_blank">隐私条款</a>
            </div>
            <div class="col-xs-6 col-sm-3 col-md-2 footer-col" style="color:darkgrey;">
                <div class="col-title" style="color:white;" >综合实训</div>
                <a href="#a" target="_blank">人文社科</a><br>
                <a href="#a" target="_blank">自然科学</a><br>
                <a href="#a" target="_blank">计算机</a><br>
                <a href="#a" target="_blank">全部课程</a>
            </div>
        </div>
    </div>
</div>



















<!--模态框div-->
<div class="modal fade" id="signup" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-sm" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal" aria-hidden="true" >
                    &times;
                </button>
                <h4 class="modal-title" id="myModalLabel" style="text-align:center;font-size:22px;color:black;">
                    注册
                </h4>
            </div>
            <div class="modal-body" >
                <form class="form-group" action="studentRegiste" method="post" id="registerForm" target="hidden_frames">
                    <iframe name="hidden_frames" style="display:none;"></iframe>
                    <div class="form-group">
                        <label >账号</label>
                        <input class="form-control" type="text" placeholder="6-15位字母或数字" name="studentAccount" id="studentAccount"/>
                    </div>
                    <div class="form-group">
                        <label >名字</label>
                        <input class="form-control" type="text" placeholder="给自己起个名吧" name="studentName" id="studentName"/>
                    </div>
                    <div class="form-group">
                        <label >密码</label>
                        <input class="form-control" type="password" placeholder="至少6位字母或数字" name="studentPassword" id="studentPassWord"/>
                    </div>
                    <div class="form-group">
                        <label >再次输入密码</label>
                        <input class="form-control" type="password" placeholder="至少6位字母或数字" id="studentPasswordAgain"/>
                    </div>
                    <div class="form-group">
                        <label >邮箱</label>
                        <input class="form-control" type="email" placeholder="例如:123@123.com" name="studentEmail" id="studentEmail"/>
                    </div>
                    <div class="form-group">
                        <label >性别</label>
                        <input type="radio" name="studentSex" value="男"/>男&nbsp&nbsp&nbsp
                        <input type="radio" name="studentSex" value="女" checked />女
                    </div>
                    <div class="form-group">
                        <input type="hidden" name="createDate" id="createDate"/>
                    </div>
                    <div class="text-right">
                        <button class="btn btn-primary" type="submit" id="register">注册</button>
                        <button class="btn btn-danger" data-dismiss="modal">取消</button>
                    </div>
                    <a href="" data-toggle="modal" data-dismiss="modal" data-target="#signin">已有账号？点我登录</a>
                </form>
            </div>
        </div>
    </div>
</div>
<div class="modal fade" id="signin" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-sm" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal" aria-hidden="true" >
                    &times;
                </button>
                <h4 class="modal-title" id="myModalLabel1" style="text-align:center;font-size:22px;color:black;">
                    登录
                </h4>
            </div>
            <div class="modal-body" >
                <form class="form-group"  method="post" id="loginForm">
                    <div class="form-group">
                        <label >用户名</label>
                        <input class="form-control" type="text" placeholder="" id="account" name="account">
                    </div>
                    <div class="form-group">
                        <label >密码</label>
                        <input class="form-control" type="password" placeholder="" id="password" name="password">
                    </div>
                    <div class="form-group">
                        <label >身份:</label>

                        <input type="radio" value="student" id="studentRadio" name="accountType"  checked />学生 &nbsp&nbsp&nbsp
                        <input type="radio" value="teacher" id="teacherRadio" name="accountType"    />老师

                    </div>
                    <div class="text-right">
                        <button class="btn btn-primary" type="submit" onclick="return Login();">登录</button>
                        <button class="btn btn-danger" data-dismiss="modal">取消</button>
                    </div>
                    <a href="" data-toggle="modal" data-dismiss="modal" data-target="#signup">没有账号？点我注册</a>
                </form>
            </div>
        </div>
    </div>
</div>




<div class="modal fade" id="searchCourseResult" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-sm" role="document" style="width:80%;height:auto;">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal" aria-hidden="true" >
                    &times;
                </button>
            </div>
            <div class="modal-body" >
                <div style="width:100%;height:auto;" id="searchResultDisplay">
                    <div style="width:100%;height:230px;" class="searchResultRow"><!--搜索课程结果一行-->
                        <div style="width:30%;margin-left:2%;height:210px;float:left;" class="oneSearchCourse" >
                            <div style="width:100%;height:160px;background-image:url('img/1.jpg');background-repeat:no-repeat;background-size:cover;">
                                <span style="background-color:rgba(0,0,0,0.4);margin-left:1px;margin-top:3px;color:white;font-size:15px;width:auto;height:auto;">正在进行</span>
                            </div>
                            <div class="roughIntro" style="font-size:17px;width:100%;height:50px;margin-top:5px;" >
                                <div>
                                    <span style="color:black;font-size:21px;">计算机组成原理</span>
                                </div>
                                <div>
                                    北京大学 张教授 <span style="font-size:17px;margin-left:15px;color:grey;">3433</span><span style="font-size:17px;margin-left:10px;color:grey;">556</span>
                                </div>
                            </div>
                            <div class="detailIntro" style="display:none;width:100%;height:50px;color:darkgray;background-color:rgb(250, 240, 240);overflow:hidden;">awioehgirhihauifhrgjaosdkijofiahrgfhksndfhgauiergsjefrhgiurfhseiuhgsjzfiudhasiwurgbrij</div>
                        </div>
                    </div><!--搜索课程结果一行-->
                </div>
            </div>
        </div>
    </div>
</div>
</body>
</html>
<script type="text/javascript">
    function show(id) {
        var objDiv = $("#"+id+"");

        $(objDiv).css("display","block");

        $(objDiv).css("left", event.clientX);

        $(objDiv).css("top", event.clientY + 10);

    }

    function hide(id) {
        var objDiv = $("#"+id+"");
        $(objDiv).css("display", "none");
    }
    function details(){
        $("div.rightbmg #logocomorsch").mouseover(function(){
            var toph=$(document).scrollTop();
            var to=120-toph;
            $("div.rightbmg p").css("top",to);
            $("div.rightbmg p").slideToggle("fast");
        });
        $("div.rightbmg #logocomorsch").mouseout(function(){
            $("div.rightbmg p").css({"display":"none"});
        });
        $(".coursebgm").mouseover(function(e){
            var bgmid=e.target.id;
            var num=bgmid.substring(9,11);
            var introid="course-intro"+num;
            //if(e.target.id==("course"))
            $("#"+introid).fadeIn("slow");
        })
        $(".coursebgm").mouseout(function(){

            $(".course-intro").css({"display":"none"});
        });
    }
    $(document).ready(details);
    function getUrlParam(name) {
        var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)"); //构造一个含有目标参数的正则表达式对象
        var r = window.location.search.substr(1).match(reg);//匹配目标参数
        if (r != null) {
            var ss = unescape(r[2]);
            return ss;
        }
        var ss = "course";
        return ss; //返回参数值
    }
    var teacherId = "<%=teacherId%>";
    var studentId = "<%=studentId%>";
    $(document).ready(function () {
        $("#turnDisPlay").show();
        var sch=[];
        var com=[];
        var inserts = getUrlParam("insert");
        if (inserts == "true") {
            alert("注册成功!");
        }
        else if (inserts == "false" || inserts == "errorAndException") {
            alert("注册失败!");

        }
        if ( studentId == "-1")
        {
            if ( teacherId == "-1")
            {
                $("#beforeLogin").show();
                $("#afterLogin").hide();
            }
            else {
                $("#beforeLogin").hide();
                $("#afterLogin").show();
                var teacherPic = "img/" + "<%=teacherPic%>";
                $("#Pic").attr("src", teacherPic);
                $("#afterLogin").attr("title", "<%=teacherName%>");
            }
        }
        else {
            $("#beforeLogin").hide();
            $("#afterLogin").show();
            var studentPic = "img/" + "<%=studentPic%>";
            $("#Pic").attr("src", studentPic);
            $("#afterLogin").attr("title", "<%=studentName%>");
            $.ajax({
                type: "GET",
                data: { "studentId": studentId },
                url: "getReCommandCourse?t=" + new Date().getTime(),
                dataType: "json",
                cache: false,
                success: function (data) {
                    if(data.length>0){
                        $("#recommandCourses").show();
                        $.each(data, function (i, values) {
                            var coursehref = "#coursehref" + (i + 13);
                            var coursebgm = "#coursebgm" + (i + 13);
                            var coursestatus = "#course-status" + (i + 13);
                            var coursefooterhref = "#course-footerhref" + (i + 13);
                            var coursetitle = "#course-title" + (i + 13);
                            var courseauthor = "#course-author" + (i + 13);
                            var fans = "#fans" + (i + 13);
                            var comments = "#comments" + (i + 13);
                            var courseintro = "#course-intro" + (i + 13);
                            $(coursehref).attr("href", "courseIntroduce.jsp?courseid=" + values["courseId"]);
                            $(coursefooterhref).attr("href", "courseIntroduce.jsp?courseid=" + values["courseId"]);
                            $(coursebgm).css({
                                backgroundImage: 'url("img/' + values["courseImgSrc"] + '")'
                            });
                            $(coursestatus).text(values["courseState"]);
                            $(coursetitle).text(values["courseName"]);
                            $(courseauthor).text(values["school"] + " " + values["teachers"]);
                            $(fans).text(values["fans"]);
                            $(comments).text(values["comments"]);
                            $(courseintro).text(values["courseIntroduce"]);
                            $("#content" + (i + 1)).show();
                        })
                    }

                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            })

        }
        $.ajax({
            type:"GET",
            url:"indexCourse?t="+new Date().getTime(),//首页课程
            cache:false,
            dataType:"json",
            success:function(data){
                $.each(data,function(i,values){
                    var coursehref="#coursehref"+(i+1);
                    var coursebgm="#coursebgm"+(i+1);
                    var coursestatus="#course-status"+(i+1);
                    var coursefooterhref="#course-footerhref"+(i+1);
                    var coursetitle="#course-title"+(i+1);
                    var courseauthor="#course-author"+(i+1);
                    var fans="#fans"+(i+1);
                    var comments="#comments"+(i+1);
                    var courseintro="#course-intro"+(i+1);
                    $(coursehref).attr("href","courseIntroduce.jsp?courseid="+values["courseId"]);
                    $(coursefooterhref).attr("href","courseIntroduce.jsp?courseid="+values["courseId"]);
                    $(coursebgm).css({
                        backgroundImage:'url("img/'+values["courseImgSrc"]+'")'
                    });
                    $(coursestatus).text(values["courseState"]);
                    $(coursetitle).text(values["courseName"]);
                    $(courseauthor).text(values["school"]+" "+values["teachers"]);
                    $(fans).text(values["fans"]);
                    $(comments).text(values["comments"]);
                    $(courseintro).text(values["courseIntroduce"]);
                })
            },
            error:function(e){
                alert(e.responseText);
            }
        })
        $.ajax({
            type:"GET",
            url:"indexSch?t="+new Date().getTime(),//首页展示合作院校
            cache:false,
            dataType:"json",
            success:function(data){
                $.each(data,function(i,val){
                    var j=i+1;
                    if(j<=8){
                        var comhref="#comhref"+j;
                        var comimg="#comimg"+j;
                        var comname="#comname"+j;
                        $(comhref).attr("href","companyLogin.jsp?comorschid="+val.schoolid);
                        $(comimg).attr("src","img/"+val.pic);
                        $(comname).text(val.schoolname);
                    }
                })
            },
            error:function(e){
                alert(e.responseText);
            }

        })
        var count=0;
        $('#signup').on('show.bs.modal', function (e) {
            // 关键代码，如没将modal设置为 block，则$modala_dialog.height() 为零
            $(this).css('display', 'block');
            var modalHeight=$(window).height() / 32 - $('#youModel .modal-dialog').height() ;
            $(this).find('.modal-dialog').css({
                'margin-top': modalHeight
            });
        });
        $('#signin').on('show.bs.modal', function (e) {
            // 关键代码，如没将modal设置为 block，则$modala_dialog.height() 为零
            $(this).css('display', 'block');
            var modalHeight=$(window).height() / 20 - $('#youModel .modal-dialog').height() ;
            $(this).find('.modal-dialog').css({
                'margin-top': modalHeight
            });
        });
        $("#loginbutton").click(function(){
            window.location.href="faceLoginStuAndTeach.jsp";
        })
        $("#myCarousel").carousel({
            interval: 2000
        })
        $("#myCarousel1").carousel({
            interval: 2000
        })
        //$("#myCarousel").carousel("cycle");
        $("#myCarousel").carousel("prev");
        $("#myCarousel1").carousel("prev");
        $("#viewallsch").click(function(){
            window.location.href="allcomOrSch.html";
        })
    })
    $(document).on("click", "#ownIndex", function () {
        if (studentId!="-1") {
            window.location.href="students.jsp";
        }
        else {
            if (teacherId != "-1") {
                window.location.href = "teachers.jsp";
            }
            else {
                alert("a unexpected error happened!");
            }
        }
    });
    $(document).on("click", "#personalInfo", function () {
        if (studentId!="-1") {
            window.location.href="studentInfo.jsp";
        }
        else {
            if (teacherId != "-1") {
                window.location.href = "teacherInfo.jsp";
            }
            else {
                alert("a unexpected error happened!");
            }
        }
    });
    $(document).on("click", "#exit", function () {
        $.ajax({
            type: "GET",
            url: "exitSystem?t=" + new Date().getTime(),
            data: { "exit": "true" },
            dataType: "json",
            cache: false,
            success: function (data) {
                window.location.href="./index.jsp";
            },
            error: function (xhr) {
                alert(xhr.responseText);
            }
        });
    });
    $(function () {
        $("[data-toggle='popover']").popover({
            trigger: 'manual',
            placement: 'bottom', //placement of the popover. also can use top, bottom, left or right
            html: 'true', //needed to show html of course
            content: '<a  id="ownIndex">个人主页</a><hr><a  id="personalInfo">个人信息</a><hr><a  id="exit">退出</a>',
            animation: false
        }).on("mouseenter", function () {
            var _this = this;
            $(this).popover("show");
            $(this).siblings(".popover").on("mouseleave", function () {
                $(_this).popover('hide');
            });
        }).on("mouseleave", function () {
            var _this = this;
            setTimeout(function () {
                if (!$(".popover:hover").length) {
                    $(_this).popover("hide");
                }
            }, 100);
        });
    });
    function Login() {
        if ($("#account").val() != ""&& $("#password").val() != "") {
            if ($('input[name="accountType"]:checked ').val() == "teacher") {
                $("#account").attr("name","teacherAccount");
                $("#password").attr("name","teacherPassword");
                $("#loginForm").attr("action", "teacherLogin");
                return true;
            }
            else if ($('input[name="accountType"]:checked ').val() == "student") {
                $("#account").attr("name","studentAccount");
                $("#password").attr("name","studentPassword");
                $("#loginForm").attr("action", "studentLogin");
                return true;
            }
        }
        alert("账号密码为必填项!");
        return false;
    }
    $(document).on("mouseover", ".courseBac", function () {
        var id1 = $(this).attr("id").substr(9, 3);
        $("#roughIntro"+id1).hide();
        $("#detailIntro"+id1).fadeIn("slow");

    })
    $(document).on("mouseout", ".courseBac", function () {

        $(".detailIntro").hide();
        $(".roughIntro").fadeIn("slow");

    })
    function getNowFormatDate() {
        var date = new Date();
        var seperator1 = "-";
        var year = date.getFullYear();
        var month = date.getMonth() + 1;
        var strDate = date.getDate();
        if (month >= 1 && month <= 9) {
            month = "0" + month;
        }
        if (strDate >= 0 && strDate <= 9) {
            strDate = "0" + strDate;
        }
        var currentdate = year + seperator1 + month + seperator1 + strDate;
        return currentdate;
    }
    $(document).on("click", "#register", function () {
        var createDate="";
        createDate=getNowFormatDate();
        $("#createDate").val(createDate);
        if ($("#studentAccount").val()=="") {
            alert("账号为空!");
            return false;
        }
        if ($("#studentName").val() == "") {
            alert("名字为空!");
            return false;
        }
        if ($("#studentPassWord").val() == "") {
            alert("密码为空!");
            return false;
        }
        if ($("#studentPassWordAgain").val() != $("studentPassWord").val()) {
            alert("两次密码不一致，请检查!");
            return false;
        }
        return true;
    })
</script>
<script >
    $(function () {
        $(".searchbutton").click(function () {
            if ($("#searchText").val() == "" || $("#searchText").val() == "在这里搜索课程") {
                alert("请输入关键字信息");
            }
            else {
                $.ajax({
                    type: "GET",
                    url: "searchCourse?t=" + new Date().getTime(),
                    data: { "courseName": $("#searchText").val() },
                    cache: false,
                    dataType: "json",
                    success: function (data) {
                        if (data.length == 0) {
                            $("#searchResultDisplay").empty();
                            $("#searchResultDisplay").html('<div><img src="img/nonesearch1.jpg" style="width:160px;margin-left:20%;"><span style="width:60%;margin-left:20px;;font-size:22px;color:black;">暂时没有你想要的课程~~~</span></div>');
                        }
                        else {
                            var rows = parseInt(data.length / 3);
                            var last = data.length - rows * 3;
                            var str1 = "";
                            $("#searchResultDisplay").empty();
                            for (var i = 0; i < (rows + 1) ; i++) {
                                str1 = ' <div style="width:100%;height:230px;" class="searchResultRow" id="searchResultRow' + (i + 1) + '"></div>';
                                $("#searchResultDisplay").append(str1);
                                var str2 = "";
                                $("#searchResultRow" + (i + 1)).empty();
                                if (i < rows) {
                                    for (var j = 0; j < 3; j++) {
                                        str2 = '<div style="width:30%;margin-left:2%;height:210px;float:left;" class="oneSearchCourse" id="oneSearchCourse' + data[3 * i + j].courseId + '">' +
                                            '<a href="courseIntroduce.jsp?courseid=' + data[3 * i + j].courseId + '">' + '<div style="width:100%;height:160px;background-image:url(' + "'" + "./img/" + data[3 * i + j].courseImgSrc + "'" + ');background-repeat:no-repeat;background-size:cover;" class="courseBac" id="courseBac' + data[3 * i + j].courseId + '">' +
                                            '<span style="background-color:rgba(0,0,0,0.4);margin-left:1px;margin-top:3px;color:white;font-size:15px;width:auto;height:auto;">' + data[3 * i + j].courseState + "</span>" +
                                            " </div></a>" +
                                            ' <div class="roughIntro" id="roughIntro' + data[3 * i + j].courseId + '"style="font-size:17px;width:100%;height:50px;margin-top:5px;" >' +
                                            " <div>" +
                                            '<span style="color:black;font-size:21px;">' + data[3 * i + j].courseName + "</span>" +
                                            "</div>" +
                                            "<div>" +
                                            data[3 * i + j].school + "&nbsp&nbsp" + data[3 * i + j].teachers + '<span style="font-size:17px;margin-left:15px;color:grey;">'+data[3*i+j].fans+'</span><span style="font-size:17px;margin-left:10px;color:grey;">'+data[3*i+j].comments+"</span>" +
                                            "</div>" +
                                            "</div>" +
                                            '<div class="detailIntro" id="detailIntro' + data[3 * i + j].courseId + '" style="display:none;width:100%;height:50px;color:darkgray;background-color:rgb(250, 240, 240);overflow:hidden;">' + data[3 * i + j].courseIntroduce + "</div>" +
                                            "</div>";
                                        $("#searchResultRow" + (i + 1)).append(str2);
                                    }
                                }
                                else {
                                    for (var j = 0; j < last; j++) {
                                        str2 = '<div style="width:30%;margin-left:2%;height:210px;float:left;" class="oneSearchCourse" id="oneSearchCourse' + data[3 * i + j].courseId + '">' +
                                            '<a href="courseIntroduce.jsp?courseid=' + data[3 * i + j].courseId + '">' + '<div style="width:100%;height:160px;background-image:url(' + "'" + "img/" + data[3 * i + j].courseImgSrc + "'" + ');background-repeat:no-repeat;background-size:cover;" class="courseBac" id="courseBac' + data[3 * i + j].courseId + '">' +
                                            '<span style="background-color:rgba(0,0,0,0.4);margin-left:1px;margin-top:3px;color:white;font-size:15px;width:auto;height:auto;">' + data[3 * i + j].courseState + "</span>" +
                                            " </div></a>" +
                                            ' <div class="roughIntro" id="roughIntro' + data[3 * i + j].courseId + '"style="font-size:17px;width:100%;height:50px;margin-top:5px;" >' +
                                            " <div>" +
                                            '<span style="color:black;font-size:21px;">' + data[3 * i + j].courseName + "</span>" +
                                            "</div>" +
                                            "<div>" +
                                            data[3 * i + j].school + "&nbsp&nbsp" + data[3 * i + j].teachers + '<span style="font-size:17px;margin-left:15px;color:grey;">'+data[3*i+j].fans+'</span><span style="font-size:17px;margin-left:10px;color:grey;">'+data[3*i+j].comments+"</span>" +
                                            "</div>" +
                                            "</div>" +
                                            '<div class="detailIntro" id="detailIntro'+data[3*i+j].courseId+'" style="display:none;width:100%;height:50px;color:darkgray;background-color:rgb(250, 240, 240);overflow:hidden;">' + data[3 * i + j].courseIntroduce + "</div>" +
                                            "</div>";
                                        $("#searchResultRow" + (i + 1)).append(str2);
                                    }

                                }
                            }
                        }

                        $("#searchCourseResult").modal("show");
                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                    }
                })
            }

        })
    })</script>
