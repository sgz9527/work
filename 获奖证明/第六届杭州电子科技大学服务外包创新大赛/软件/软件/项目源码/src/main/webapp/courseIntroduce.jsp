<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/1/14 0014
  Time: 14:50
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
<html xmlns="http://www.w3.org/1999/xhtml">
<head >
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>课程介绍</title>
    <link href="css/courseintroduce.css" rel="stylesheet" type="text/css">
    <link rel="stylesheet" href="./css/bootstrap.min.css">
    <script src="js/jquery-2.0.3.js"></script>
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
    <script>
        var courseids;
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
        $(document).ready(function () {
            courseids = getUrlParam("courseid");
            /***************************************************************/
            $.ajax({
                type: "GET",
                url: "getCourse?t=" + new Date().getTime(),
                data:{ "courseid":courseids },
                dataType: "json",
                cache: false,
                success: function (data) {
                    var strhtml2='<div style="width:auto;height:auto;margin-left:60px"><video width="410px" height="295px"  controls preload="auto" poster="'+"./img/"+data["courseImgSrc"]+'"'+
                        'data-setup="{}">'+'<source src="'+"./video/"+data["introduceVideoSrc"]+'" type="video/mp4" />'+"</video></div>";
                    $(".introducevideo").empty();
                    // alert(strhtml);
                    $(".introducevideo").html(strhtml2);
                    $("#courseName").empty();
                    $("#courseName").html("<strong>" + data["courseName"] + "</strong>");
                    $("#chapterNum").text(data["chapterNum"]);
                    $("#courseProgress").text(data["courseProgress"]);
                   // alert("isChoose:"+data.isChoose+" isManage:"+data.isManage);
                    if (data["isChoose"] == "true") {
                        $("#alreadyChoose").show();
                        $("#notChoose").hide();
                        $("#alreadyManage").hide();
                    }
                    else if (data["isManage"] == "true") {
                        $("#alreadyManage").show();
                        $("#notChoose").hide();
                        $("#alreadyChoose").hide();
                    }
                    else {
                        $("#notChoose").show();
                        $("#alreadyChoose").hide();
                        $("#alreadyManage").hide();
                    }

                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            })
            $.ajax({
                type: "GET",
                url: "/getTeacher?t=" + new Date().getTime(),
                data: { "courseid": courseids },
                dataType: "json",
                cache: false,
                success: function (data) {
                    $(".teacherslist").empty();
                    $.each(data, function (i, value) {
                        var strhtm = '<div class="teacher">' +
                            '<div class="teacherimg">' +
                            '<img src="./img/' + value["teacherPic"] + '" style="height:inherit;border-radius:35px;">' +
                            "</div>" +
                            '<div class="teachername"><span style="float:left;width:40%;color:black;margin-top:5%;">' + value["teacherName"] + "</span>" +
                            '<span style="color:darkgrey;margin-top:5%;float:left;width:40%;">' + value["teacherLevel"] + "</span>"
                            +"</div>" +
                            "</div>";
                        $(".teacherslist").append(strhtm);
                    })
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            })
        })
        $(document).on("click", "#goToStudy", function () {
            if(studentId!="-1")
                window.open("study.jsp?courseid="+courseids);
        });
        $(document).on("click", "#goToManage", function () {
            // window.location.href = "study.jsp";
            if ( teacherId != "-1")
                window.open("courseManage.jsp?courseid=" + courseids);
        });
        $(document).on("click", "#chooseCourse", function () {
            if(teacherId!="-1")
            {
                alert("forbidden account:teacher cannot choose course");
                return false;
            }
            if ( studentId == "-1") {
                $("#signin").modal("show");
            }
            else {
                $.ajax({
                    type: "POST",
                    url: "/chooseCourse?t=" + new Date().getTime(),
                    data: { "courseid": courseids},
                    dataType: "json",
                    cache: false,
                    success: function (data) {
                        if(data.result=="ok")
                        {
                            alert("已成功报名课程");
                            window.location.reload(true);
                        }
                        else alert("fail to choose this course");
                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                                alert("unknown error");
                    }
                });
            }

        });
        $(function () {
            $("[data-toggle='popover']").popover({
                trigger: 'manual',
                placement: 'bottom', //placement of the popover. also can use top, bottom, left or right
                title: '<div style="text-align:center; color:red; text-decoration:underline; font-size:14px;"> Muah ha ha</div>', //this is the top title bar of the popover. add some basic css
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
            $(document).on("click", "#ownIndex", function () {
                if ( studentId != "-1") {
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
                    dataType: "json",
                    cache: false,
                    success: function (data) {
                        if(data.exit=="ok")
                        window.location.href = "index.jsp";
                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                    }
                });
        });
    </script>
    <script src="js/login.js">
    </script>
    <style>
        .vjs-default-skin .vjs-big-play-button {left:40%;top:40%;}
    </style>
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
            <input class="text" value="在这里搜索课程" onfocus="if (value =='在这里搜索课程'){value =''}" onblur="if (value ==''){value='在这里搜索课程'}" />
            <input type="button" class="searchbutton" value="搜索" id="button1">
        </div>
        <div style="width:20%;float:left;height:inherit;padding-top:8px;padding-right:0px;" id="beforeLogin">
            <a data-target="#signin" data-toggle="modal"  >登录</a>
            <a data-target="#signup" data-toggle="modal" >注册</a>
        </div>
        <div style="width:20%;float:left;height:inherit;padding-top:4px;padding-right:0px;display:none;" id="afterLogin" data-toggle="popover" title="???" data-placement="bottom" >
            <img src="." style="height:30px;width:30px;border-radius:15px;border-style:none;" id="Pic"/>
        </div>
    </div>
</div>
<div class="title">首页->全部课程->外语</div>
<span style="color:black;font-size:23px;margin-left:30px;"><strong>播放课程简介</strong></span><img src="img/t014bbb00fd69c0fe5f.png" style="height:27px;">
<div class="introduce">
    <div class="introducevideo">
        <!--<video class="video-js vjs-default-skin" id="videoPost" controls preload="auto" width="485" height="inherit"  poster=""
    data-setup="{}" >
    </video>-->
    </div>
    <div class="rightofintroduce">
        <span style="color:black;font-size:28px;display:block;" id="courseName"></span>
        <div class="time">
            <div><span>开课时间：</span><span>2017年12月23日</span></div>
            <div><span>结课时间:</span><span>2018年12月23日</span>
                <span style="margin-left:215px;color:red;">进行到第</span><span style="color:red;" id="courseProgress"></span><span style="color:red;">章&nbsp&nbsp&nbsp共</span><span style="color:red;" id="chapterNum"></span><span style="color:red;">章</span>
            </div>
            <span>课时安排:</span><span>12</span>课时
        </div>
        <div style="color:darkgrey;margin-top:15px;"><span style="color:orange;"><strong>6666</strong></span>人已参加学习</div>
        <div id="notChoose">
            <input type="button" value="立即报名学习" style="color:white;background:rgba(93,228,37,1.00);width:190px;height:60px;font-size:19px;" id="chooseCourse">
            <span style="display:block;margin-top:0px;font-size:16px;color:black;">怕错过精彩内容？</span><span style="color:orange;font-size:16px;margin-top:15px;">报名下一次课</span>
        </div>
        <div id="alreadyChoose" style="display:none;">
            <input type="button" value="已参加，进入学习" style="color:white;background:rgba(93,228,37,1.00);width:190px;height:60px;font-size:19px;" id="goToStudy">
        </div>
        <div id="alreadyManage" style="display:none;">
            <input type="button" value="进入管理" style="color:white;background:rgba(93,228,37,1.00);width:190px;height:60px;font-size:19px;" id="goToManage">
        </div>
    </div>
</div>
<div class="moredetail">
    <div class="moreinfo">
        <div class="infotitle">
            <span style="color:black;font-size:21px;margin-left:57px;float:left;">课程详情</span >
            <div style="float:left;width:auto;height:auto;margin-left:30px;" >
                <span style="color:black;font-size:21px;margin-left:90px;">课程评价(</span>
                <span style="font-size:21px;">71</span>
                <span style="font-size:21px;">)</span>
            </div>
        </div>


        <!--课程详情部分开始-->
        <div class="coursedetail">
            <div class="team">
                <p style="font-size:14px;color:black;">世界那么大，想出去看看，不会英语怎么行？带字典，太沉啦！请翻译，太土豪！ 英语那么难，想通过自考，没有老师怎么行？去上课，没时间！请家教，太奢侈！ 你有木有感到压（hao）力（wu）山（ban）大（fa）（怒摔 QAQ~~） 别怕，贝壳（北科）来帮忙！ 理解课文学习词汇分析难句品味文化- 帮你全搞定！</p>
                <div style="width:100%;height:auto;"><span style="float:right;font-size:16px;color:black;">-------课程团队</span>
                </div>
            </div>
            <span style="display:block;font-size:22px;color:black;margin-left:55px;margin-top:30px;"><strong>课程介绍</strong></span>


            <div class="introducedetail">
                <p>本课程共包括六个主题单元，每个单元围绕一个主题展开，包括两篇课文。授课视频化整为零，让英语学习系统、灵活、高效。在每篇课文的视频课程里，有课文逐句讲解、单词拓展延伸、难句分析总结、文化对比欣赏，更有精彩生动的课文导入和课文点睛，助力广大英语学习者通过自学提升综合应用英语的能力。你的学习之旅将是酱紫滴：</p>

                <h5>	s   Relax and take it slow. </h5>
                <p>老师带你一起熟悉课文主题，讲个故事，聊聊经历，说说感受，让你紧绷的精神放松下来，让英语慢慢走近你的世界，走近你的心里。</p>
                <h5>s   What’s it all about? </h5>
                <p>老师为你逐句讲解课文，有英文，有翻译，有讲解，再来点儿单词的拓展，风趣的点评；不用怕跟丢，因为讲到哪句亮哪句；别担心记不住，因为总有小测等着你。妥妥的。</p>
                <h5>s   Let’s learn some new words.</h5>
                <p>
                    学好英文单词老难了！老师说啦，学单词这事儿啊，跟做饭一样一样的，买菜（积累单词），做菜（分析单词），开吃（应用单词）！有些单词意思太多，咱就学最常用的，最爱考的；有些单词变化多端，咱就分析词根，分析词缀；有些单词学完就忘，咱就做几道小题，强化记忆；再配上有趣的例句和原创的美图，跟着老师一起，饕餮一番吧！</p>
                <h5>s   Long sentences? Break them down!</h5>
                <p>有些个英文句子，每个单词都认识，放到一块就蒙圈。别急啊，老师这不是来了吗？主谓宾、定状补，全都讲得到，一个跑不了。</p>
                <h5>s   Take a sip of culture. </h5>
                <p>语言和文化，文化和语言，缠缠绵绵不分离。了解点儿西方文化，知道点儿文化差异，学英语才上档次。</p>
                <h5>s   Let’s wrap it up! </h5>
                <p>跟着老师一起品尝一下“征服英语”的成就感吧。学完一篇课文，就意味着你的英语又提高了一点儿，你离目标又进了一步。如果你也想跟老师一样侃侃而谈，激扬文字，那就呼朋唤友，到论坛里小试牛刀、直抒胸臆吧。</p>
            </div>

            <div class="category">
                课程大纲

            </div>
            <div>
                预备知识
            </div>
            <div>
                证书要求
            </div>
            <div>
                常见问题
            </div>
        </div>
        <!--课程详情部分结束-->














        <!--课程评价部分开始-->



        <!--课程评价部分结束-->


    </div>
    <div class="teachers">
        <div
                style="padding-bottom:20px;border-bottom-style:solid;border-bottom-color:rgba(245,237,237,1.00);width:100%;height:auto;">
            <img src="./img/t0134e93e0e289548e8.png" style="margin-left:15%;margin-right:15%;width:20%;height:4%;" >
        </div>
        <div class="teacherstitle">
            <span style="text-align:left;font-size:23px;"><strong style="color:limegreen">|</strong></span>
            <span style="color:darkgrey;font-size:21px;margin-left:3px;"><strong>授课老师</strong></span>
        </div>
        <div class="teacherslist">
        </div>
    </div>
</div>
<!--模态框-->
<div class="modal fade" id="signup" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-sm" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal" aria-hidden="true" id="signUpButton">
                    &times;
                </button>
                <h4 class="modal-title" id="myModalLabel" style="text-align:center;font-size:22px;color:black;">
                    注册
                </h4>
            </div>
            <div class="modal-body" >
                <form class="form-group" action="../个人主页/courseManage.aspx" method="post">
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
                <button type="button" class="close" data-dismiss="modal" aria-hidden="true" id="signInButton" >
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
</body>
</html>

