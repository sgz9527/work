<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/5/8 0008
  Time: 11:30
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.xqq.pojo.Manager" %>
<%@ page import="com.xqq.pojo.Teacher" %>
<%@ page import="com.xqq.pojo.Student" %>
<%
    Integer managerId=-1;
    String managerAccount="null";
    String managerPassword="null";
    String managerName=" ";
    boolean isStu=false;
    boolean isTeacher=false;
    Manager manager=(Manager)session.getAttribute("adminInfo");
    Teacher teacher=(Teacher)session.getAttribute("teacherInfo");
    Student student=(Student)session.getAttribute("studentInfo");
    if(teacher!=null){
        isTeacher=true;
    }else if(student!=null){
        isStu=true;
    }
    if(manager!=null){
        managerId=manager.getManagerId();
        managerAccount=manager.getAccount();
        managerPassword=manager.getPassword();
        managerName=manager.getManagerName();
    }
%>
<html>
<head>
    <title>管理员工作界面</title>
    <link href="css/bootstrap.min.css"  rel="stylesheet" type="text/css">
    <script src="js/jquery-2.0.3.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <link href="css/main.css" rel="stylesheet">
</head>
<body>
<!--顶部导航栏部分-->
<nav class="navbar navbar-inverse">
    <div class="container-fluid">
        <div class="navbar-header">
            <a class="navbar-brand" title="logoTitle" href="#">logo</a>
        </div>
        <div class="collapse navbar-collapse">
            <ul class="nav navbar-nav navbar-right">
                <li role="presentation">
                    <a >当前用户：<span class="badge" id="adminName"></span></a>
                </li>
                <li>
                    <a><span class="glyphicon glyphicon-lock" id="exitLogin"></span>退出登录</a>
                </li>
            </ul>
        </div>
    </div>
</nav>

<!-- 中间主体内容部分
 管理员功能包括：
 1.管理首页
 2.课程管理
 3.评论审核
 5.教师管理
 6.院校管理
 -->
<div class="pageContainer">
    <!-- 左侧导航栏 -->
    <div class="pageSidebar">
        <ul class="nav nav-stacked nav-pills">
            <li role="presentation">
                <a href="changeTurnDisplay.html" target="mainFrame" >管理首页</a>
            </li>
            <li role="presentation">
                <a href="manageTeacher.html" target="mainFrame">教师管理</a>
            </li>
            <li role="presentation">
                <a href="manageSch.html" target="mainFrame">院校管理</a>
            </li>
            <li role="presentation">
                <a href="manageCourse.html" target="mainFrame">课程管理</a>
            </li>
            <li role="presentation">
                <a href="checkComment.html" target="mainFrame">评论审核</a>
            </li>
            <!-- 开始 -->
            <li class="dropdown">
                <a class="dropdown-toggle" data-toggle="dropdown" href="nav4.html" target="mainFrame">
                    个人设置<span class="caret"></span>
                </a>
                <ul class="dropdown-menu">
                    <li>
                        <a  target="mainFrame">修改密码</a>
                    </li>
                    <li>
                        <a  target="mainFrame">退出系统</a>
                    </li>
                    <li>
                        <a target="mainFrame">查看个人信息</a>
                    </li>
                </ul>
            </li>
            <!-- 结束 -->
        </ul>
    </div>

    <!-- 左侧导航和正文内容的分隔线 -->
    <div class="splitter"></div>
    <!-- 正文内容部分 -->
    <div class="pageContent">
        <iframe src="welcome.jsp" id="mainFrame" name="mainFrame"
                frameborder="0" width="100%"  height="100%" frameBorder="0">
        </iframe>
    </div>

</div>
<!-- 底部页脚部分 -->
<div class="footer">

</div>
</body>
</html>
<script type="text/javascript">
    var managerId="<%=managerId%>";
    var managerName="<%=managerName%>";
    var isStu=<%=isStu%>;
    var isTeacher=<%=isTeacher%>;
    $(document).ready(function(){
        if(isStu||isTeacher){
            window.location.href="index.jsp";
        }
        if(managerId=="-1"){
            window.location.href="adminLogin.jsp";
        } else{
            $("#adminName").html(managerName);
        }
        $(".nav li").click(function() {
            $(".active").removeClass('active');
            $(this).addClass("active");
        });
        $("#exitLogin").click(function() {
            $.ajax({
                url:"exitSystem",
                dataType:"json",
                cache:false,
                type:"GET",
                success:function(data){
                    if(data.exit=="ok")
                        window.location.href="adminLogin.jsp";
                },
                error:function(e){
                    alert(e.responseText);
                }
            })
        });
    })
</script>
