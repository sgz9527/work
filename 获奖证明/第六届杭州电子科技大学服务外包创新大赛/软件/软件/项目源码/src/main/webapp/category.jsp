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
        // System.out.println("??student !=null");
        studentId=student.getStudentId();
        studentName=student.getStudentName();
        studentPic=student.getStudentPic();
    }
    //System.out.println("nullStudent");
%>
<html >
<head >
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>课程</title>
    <link href="css/category.css" rel="stylesheet" type="text/css">
    <link rel="stylesheet" href="./css/bootstrap.min.css">
    <script src="./js/jquery-2.0.3.js"></script>
    <script type="text/javascript" src="./js/bootstrap.min.js"></script>
    <script type="text/javascript">
        var teacherId = "<%=teacherId%>";
        var studentId = "<%=studentId%>";
        $(document).ready(function(){
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
            }
        })
        $(document).on("click", "#ownIndex", function () {
            if (studentId != "-1") {
                window.location.href = "students.jsp";
            }
            else {
                if ( teacherId != "-1") {
                    window.location.href = "teachers.jsp";
                }
                else {
                    alert("a unexpected error happened!");
                }
            }
        });
    </script>
    <script src="js/function.js"></script>
    <script src="js/login.js"></script>
</head>
<body>
<div class="nav">
    <div class="head" >
        <ul >
            <li style="margin-left:1px;"><img src="./img/151946214719325.png" style="height:32px;position: relative;top:0px;background: rgba(229,219,219,1.00);"></li>
            <li >
                <a href="index.jsp" >首页</a>
            </li>
            <li >
                <a href="../资讯/information.html" >资讯</a>
            </li>
            <li >
                <a href="../实训/excercise.html" >实训</a>
            </li>
            <li >
                <a href="../企业信息/companyintroduce.html" >企业</a>
            </li>
            <li >
                <a href="../交流/contact.html" >交流</a>
            </li>
            <li><a href="adminLogin.jsp">网站管理</a></li>
        </ul>
    </div>
    <div class="rightofhead">
        <div style="width:80%;float: left;height:inherit;">
            <input class="text" value="在这里搜索课程" onfocus="if (value =='在这里搜索课程'){value =''}" onblur="if (value ==''){value='在这里搜索课程'}" id="searchText"/>
            <input type="button" class="searchbutton" value="搜索" id="button1">
        </div>
        <div style="width:20%;float:left;height:inherit;padding-top:8px;padding-right:0px;" id="beforeLogin">
            <a data-target="#signin" data-toggle="modal"  >登录</a>
            <a data-target="#signup" data-toggle="modal" >注册</a>
        </div>
        <div style="width:20%;float:left;height:inherit;padding-top:4px;padding-right:0px;display:none;" id="afterLogin" data-toggle="popover" title="???" data-placement="bottom" >
            <img src="" style="height:30px;width:30px;border-radius:15px;border-style:none;" id="Pic"/>
        </div>
    </div>
</div>










<div class="fixright">
    <img src="./img/49E89BEA3F1B3F1AC788F5F94C4A457F.png">
    <span style="color:white;font-size:20px;display: block;"><strong>本周最热课程</strong></span>
    <div class="numberone">
        <div class="leftone">
            <span style="colr:white;font-size:22px;margin-left:1px;margin-top:1px;"><strong>1</strong></span>
        </div>
        <div class="rightone">
            <div style="color:red;font-size:16px;" ><a style="color:red;" href="courseIntroduce?courseId=2"><strong>爱情心理学</strong></a></div>
            <div style="color:white;margin-top:45px;"><span style="color:orange;font-size:16px;"><strong>36666</strong></span>人在学</div>
        </div>
    </div>
    <div style="width:100%;height:auto;text-align: left;margin-left:2px;">
        <span style="color:white;font-size:16px;"><strong>2.</strong></span>
        <span><a style="color:white;font-size:18px;" href="courseIntroduce?courseId=1">探索土星上的生命</a></span>
    </div>
</div>











<div class="category">
    <div class="categorytitle">
        首页-><span style="color:darkgrey;">全部课程</span><span id="types" style="color:darkgrey;">->全部</span>
    </div>
    <div class="categories">
        <table class="type">
            <tr>
                <td id="all" >全部</td>
                <td id="techonology">工科</td>
                <td id="sience">理科</td>
                <td id="social">文科</td>
                <td id="geography">地理</td>
                <td id="computer">计算机</td>
                <td id="electric">电子信息</td>
            </tr>
            <tr>
                <td id="politic">政治</td>
                <td id="foreignLanguage">外语</td>
                <td id="ware">硬件理论</td>
                <td id="sky">天文</td>
                <td id="water">水土</td>
                <td id="BaseElec">电路基础</td>
                <td id="rulesOfComputer">算法</td>
            </tr>
        </table>
    </div>
</div>
<div class="coursestate">
    <span>全部</span>&nbsp&nbsp&nbsp&nbsp&nbsp|<span>正在进行</span>&nbsp&nbsp&nbsp&nbsp&nbsp|<span>即将开始</span>&nbsp&nbsp&nbsp&nbsp&nbsp|<span>已结束</span>
</div>
<div class="allcourse">

</div>





































<!--模态框-->
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
                <form class="form-group" action="register" method="post">
                    <div class="form-group">
                        <label for="">用户名</label>
                        <input class="form-control" type="text" placeholder="6-15位字母或数字" name="accountregi" id="accountregi">
                    </div>
                    <div class="form-group">
                        <label for="">密码</label>
                        <input class="form-control" type="password" placeholder="至少6位字母或数字" name="passwordregi" id="passwordregi">
                    </div>
                    <div class="form-group">
                        <label for="">再次输入密码</label>
                        <input class="form-control" type="password" placeholder="至少6位字母或数字" id="repasswordregi">
                    </div>
                    <div class="form-group">
                        <label for="">邮箱</label>
                        <input class="form-control" type="email" placeholder="例如:123@123.com" name="emailregi" id="emailregi">
                    </div>
                    <div class="text-right">
                        <button class="btn btn-primary" type="submit">注册</button>
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
                        <label for="">用户名</label>
                        <input class="form-control" type="text" placeholder="" id="account" name="account">
                    </div>
                    <div class="form-group">
                        <label for="">密码</label>
                        <input class="form-control" type="password" placeholder="" id="password" name="password">
                    </div>
                    <div class="form-group">
                        <label for="">身份:</label>
                        <input type="radio" value="student" id="teacherRadio" name="accountType" checked />学生&nbsp&nbsp&nbsp
                        <input type="radio" value="teacher" id="studentRadio" name="accountType"  />老师
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
<script src="js/searchCourse.js"></script>

