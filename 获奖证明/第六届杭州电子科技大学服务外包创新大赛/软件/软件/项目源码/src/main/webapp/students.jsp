<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/1/15 0015
  Time: 21:23
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.xqq.pojo.Student" %>
<%
    Student student=(Student)session.getAttribute("studentInfo");
    String  studentPic="";
    String studentName="";
    String courseCount="3";
    String belongSchName="";
    String belongClass="";
    Integer studentId=-1;
     if(student!=null)
    {
        // System.out.println("??student !=null");
        studentName=student.getStudentName();
        studentPic=student.getStudentPic();
        belongSchName=student.getBelongSchName();
        belongClass=student.getBelongClass();
        studentId=student.getStudentId();
    }
    //System.out.println("nullStudent");
%>
<html >
<head >
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>学生主页</title>
    <link href="css/ownindex.css" rel="stylesheet" type="text/css">
    <link rel="stylesheet" href="./css/bootstrap.min.css">
    <script src="./js/jquery-2.0.3.js"></script>
    <script type="text/javascript" src="./js/bootstrap.min.js"></script>
    <script>
        var messageId=[];
        var isMessageFlush=false;
        function getUrlParam(name) {
            var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)"); //构造一个含有目标参数的正则表达式对象
            var r = window.location.search.substr(1).match(reg);//匹配目标参数
            if (r != null)
            {
                var ss=unescape(r[2]);
                return ss;
            }
            var ss="course";
            return ss; //返回参数值
        }
        function al(name){
            //alert(getUrlParam((name)));
            $("#alreadycourse").html('<strong><%=courseCount%></strong>');
            var dos=getUrlParam(name);
            if(dos=="course"){
                $("#111").css("color", "orange");
                $.ajax({
                    type: "GET",
                    url: "studentCourse",
                    data: { "studentId": <%=student.getStudentId()%> },
                    dataType: "json",
                    cache: false,
                    success: function (data) {
                        var counts = 0;
                        var strhtml1 = "";
                        $(".shiftdiv").empty();
                        $("#countofmaster").html(data.length);
                        $("#allcourse").html(data.length);
                        $.each(data, function (i, values) {
                            counts++;
                            var strhtml1ss =
                                '<div class="coursedetail" id="course' + counts +
                                '">' +
                                '<div class="coursepic"><img src=' +
                                '"./img/' + values["courseImgSrc"] + '"' + 'style="height:100%;border-style:none;"></div>' +
                                '<div class="rightofcoursepic">' +
                                '<div class="coursename">' +
                                '<h3 style="color:black"><strong>' + values["courseName"] + "</strong></h3>" +
                                '<span style="background-color:rgba(0,0,0,0.4);color:white;fon-size:17px;"' + 'id="courseflag' + counts + '">' + values["courseState"] + "</span></div>" +
                                '<div class="coursedisplay">' +
                                '<div class="coursebar" id="coursebar' + values["courseId"] +
                                '">' +
                                '<div id="myprogresses' + values["courseId"] + '"' + 'style="height:inherit;background:red;width:1px;"></div></div>' +
                                '<div class="courseprogress"><span>' + values["Progress"] + "</span>/<span>" + values["chapterNum"] + "</span></div>" +
                                '<div class="courseDirectoryEntry"><a href="./courseIntroduce.jsp?courseid=' + values["courseId"] + '">进入学习</a></div>' +
                                '<div class="coursedelete" id="deletecourse'+values["courseId"]+'">删除课程</div></div></div></div>';
                            strhtml1 += strhtml1ss;
                            $(".shiftdiv").append(strhtml1ss);
                            //alert(strhtml1ss);
                            var myprogress = values["Progress"];
                            var progressall = values["chapterNum"];
                            var feiwusb = parseFloat(myprogress) / parseFloat(progressall);
                            var widths = $(".coursebar").width();
                            var myprogresswidth = parseFloat(widths) * feiwusb;
                            var myprogresswidthint = parseInt(myprogresswidth);
                            //alert(strhtml1);
                            $("#myprogresses" + values["courseId"]).css("width", myprogresswidthint + "px");
                            //alert("进度条总宽度为:"+widths+"比例为:"+feiwusb+"我的进度的宽度为"+myprogresswidthint+"实际我的进度条宽度为"+$("#myprogresses"+values["courseId"]).width());
                        })
                        $(".coursehead").show();
                        $(".coursedetail").show();
                        $(".nonecourse").hide();


                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                        $(".shiftdiv").empty();
                        $(".coursehead").hide();
                        $(".coursedetail").hide();
                        $(".nonecourse").show();
                    }

                })
                //alert($("#111");
                //var element=document.getElementById("111");
                //var oStyle1 = element.currentStyle ? element.currentStyle : window.getComputedStyle(element, null);
                //var element=document.getElementById("222");
                //var oStyle2 = element.currentStyle ? element.currentStyle : window.getComputedStyle(element, null);
                //var element=document.getElementById("111");
                //var oStyle3 = element.currentStyle ? element.currentStyle : window.getComputedStyle(element, null);
                $(".mycourseinfo").css("display","block");
                $(".mytaskinfo").css("display","none");
                $(".myexcerciseinfo").css("display","none");
                $(".myscoreinfo").css("display","none");
                $(".mycapacityinfo").css("display","none");

            }
            else if (dos == "task") {
                $(".mycourseinfo").css("display","none");
                $(".mytaskinfo").css("display","block");
                $(".myexcerciseinfo").css("display","none");
                $(".myscoreinfo").css("display","none");
                $(".mycapacityinfo").css("display","none");
                isMessageFlush=true;
            }
            else if(dos=="excercise")
            {
                $(".mycourseinfo").css("display","none");
                $(".mytaskinfo").css("display","none");
                $(".myexcerciseinfo").css("display","block");
                $(".myscoreinfo").css("display","none");
                $(".mycapacityinfo").css("display","none");
            }
            else if(dos=="score"){
                $(".mycourseinfo").css("display","none");
                $(".mytaskinfo").css("display","none");
                $(".myexcerciseinfo").css("display","none");
                $(".myscoreinfo").css("display","block");
                $(".mycapacityinfo").css("display","none");
            }
            else if(dos=="capacity"){
                $(".mycourseinfo").css("display","none");
                $(".mytaskinfo").css("display","none");
                $(".myexcerciseinfo").css("display","none");
                $(".myscoreinfo").css("display","none");
                $(".mycapacityinfo").css("display","block");
            }
        }
        $(document).ready(function () {
            $.ajax({
                type: "GET",
                url: "getFinalExamScore?t=" + new Date().getTime(),
                data: { "studentId": <%=student.getStudentId()%> },
                dataType: "json",
                cache: false,
                success: function (data) {
                    $("#capacityTBody").empty();
                    $.each(data,function(i,value){
                        var strh='<tr class="">\n' +
                            '                    <td><a href="courseIntroduce.jsp?courseid='+value.courseId+'">'+value.courseName+'</a></td>\n' +
                            '                    <td>'+value.commentScore+'</td>\n' +
                            '                    <td>'+value.testScore+'</td>\n' +
                            '                    <td>'+value.examScore+'</td>\n' +
                            '                    <td>'+value.allScore+'</td>\n' ;
                        if(value.allScore<60){
                            strh+='                    <td style="color:red;">不通过</td>\n' +
                                '                </tr>';
                        }else if(value.allScore<80){
                            strh+='                    <td style="color:orange;">良好</td>\n' +
                                '                </tr>';
                        }else {
                            strh+='                    <td style="color:green;">优秀</td>\n' +
                                '                </tr>';
                        }
                        $("#capacityTBody").append(strh);
                    })
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            })
            $.ajax({
                type: "GET",
                url: "activityValue?t=" + new Date().getTime(),
                data: { "studentId": <%=student.getStudentId()%> },
                dataType: "json",
                cache: false,
                success: function (data) {
                    $("#activityValue").html(data.activityValue);
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            })
            $.ajax({
                type: "GET",
                url: "getScore?t=" + new Date().getTime(),
                data: { "studentId": <%=student.getStudentId()%> },
                dataType: "json",
                cache: false,
                success: function (data) {
                    $.each(data,function(i,val){
                        var str="<tr>\n" +
                            '<td id="belongCourse'+i+'" ><a href="courseIntroduce.jsp?courseid='+val.courseId+'">'+val.courseName+"</a></td>\n" +
                            '<td id="scorename'+i+'">'+val.scoreTitle+"</td>\n" +
                            '<td id="teachername'+i+'">'+val.scoreType+"</td>\n";
                        if(parseFloat(val.score)>=60)
                            str+= '<td id="score'+i+'"style="color:green;">'+val.score+"</td>\n";
                        else
                            str+='<td id="score'+i+'"style="color:red;">'+val.score+"</td>\n";
                        str+= "</tr>";
                        $("#ScoreTable").append(str);
                    })
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            })
            $.ajax({
                   type: "GET",
                   url: "messageInfo?t=" + new Date().getTime(),
                   cache: false,
                   data:{"studentId":<%=studentId%>},
                   dataType: "json",
                   success: function (data) {
                       if(data.messageNum!=0) {
                           if (data.messageNum <= 99)
                               $("#messageNum").html(data.messageNum);
                           else
                               $("#messageNum").html("99+");
                           $(".messageLabel").show();
                       }
                       else $(".messageLabel").hide();
                       $(".task").empty();
                       /*   消息包括：
               1.课程开课进度
               2.提醒课程作业发布
               3.课程考试发布
               4.评论回复提醒
               5.帖子有新评论提醒
               6.作业截止提醒
               7.考试截止提醒
               10.实训发布提醒
               9.评论禁言提醒
               8.禁止发帖提醒
           */
                       messageId=[];
                       $.each(data.messageList, function (i, value) {
                           if(value.messageState==-1){
                               messageId.push(value.messageId);
                           }
                           var htmStr ="";
                           if(value.messageState==-1) {
                               switch (value.messageType) {
                                   case 1:
                                       if(value.courseProgress==1){
                                           htmStr = '<div class="oneMessage">\n' +
                                               '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                               '                <span>开课了，欢迎进入该课程学习!</span>\n' +
                                               '            </div>';
                                       }
                                       else{
                                           htmStr = '<div class="oneMessage">\n' +
                                               '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                               '                <span>进行到第'+value.courseProgress+'章了，欢迎进入该课程学习!</span>\n' +
                                               '            </div>';
                                       }
                                       $(".task").append(htmStr);
                                       break;
                                   case 2:
                                       htmStr = '<div class="oneMessage">\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布作业《' + value.testName + '》，请进入该课程作业模块完成作业!</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 3:
                                       htmStr = '<div class="oneMessage">\\n\' +\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=\'+value.courseId+\'"><span ><strong>\'+value.courseName+\'</strong></span></a>\\n\' +\n' +
                                           '                <span>发布考试《\'+value.examName+\'》，请进入该课程考试模块完成作业!</span>\\n\' +\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 4:
                                       htmStr = '<div class="oneMessage" >\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>在名为《' + value.topicName + '》的帖子下发表的评论有新的回复，请查看!</span>\n' +
                                           '                <div>\n' +
                                           '\n' +
                                           '                    <div style="color:cornflowerblue">你的评论:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.commentInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '                <div>\n' +
                                           '\n' +
                                           '                    <div style="color:mediumpurple">他人回复:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.replyInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 5:
                                       htmStr = '<div class="oneMessage">\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发表的名为《' + value.topicName + '》的帖子有新评论，请查看!</span>\n' +
                                           '                <div>\n' +
                                           '\n' +
                                           '                    <div style="color:cornflowerblue">评论详情:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.commentInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 6:
                                       htmStr = ' <div class="oneMessage">\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的作业《' + value.testName + '》在' + value.testEndTime + '即将截止，请尽快进入该课程作业模块完成作业!</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 7:
                                       htmStr = ' <div class="oneMessage">\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的考试《' + value.examName + '》在' + value.examEndTime + '即将截止，请尽快进入该课程作业模块完成作业!</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 9:
                                       htmStr = '<div class="oneMessage">\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的有关评论多次被举报违规，现已限制你的发言权，从<span style="color:rebeccapurple">' + value.forbidenBeginTime + '</span>起到<span style="color:rebeccapurple">' + value.forbidenEndTime + '</span>结束，你将无法进行发表评论活动</span>\n' +
                                           '                <div>\n' +
                                           '                    <div style="color:cornflowerblue">评论详情:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.commentInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 8:
                                       htmStr = '<div class="oneMessage">\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的有关帖子多次被举报违规，现已限制你的发帖权，从<span style="color:rebeccapurple">' + value.forbidenBeginTime + '</span>起到<span style="color:rebeccapurple">' + value.forbidenEndTime + '</span>结束，你将无法进行发表话题活动</span>\n' +
                                           '                <div>\n' +
                                           '                    <div style="color:cornflowerblue">帖子详情:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.topicInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 10:
                                       htmStr = '<div class="oneMessage">\n' +
                                           '                <span>' + value.proExcerciseTime + '发布实训《' + value.excerciseName + '》，请在综合实训一栏查看</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   default:
                                       htmStr = '';
                                       break;
                               }
                           }
                           else
                           {
                               switch (value.messageType) {
                                   case 1:
                                       if(value.courseProgress==1){
                                           htmStr = '<div class="oneMessage1">\n' +
                                               '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                               '                <span>开课了，欢迎进入该课程学习!</span>\n' +
                                               '            </div>';
                                       }
                                       else{
                                           htmStr = '<div class="oneMessage1">\n' +
                                               '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                               '                <span>进行到第'+value.courseProgress+'章了，欢迎进入该课程学习!</span>\n' +
                                               '            </div>';
                                       }
                                       $(".task").append(htmStr);
                                       break;
                                   case 2:
                                       htmStr = '<div class="oneMessage1">\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布作业《' + value.testName + '》，请进入该课程作业模块完成作业!</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 3:
                                       htmStr = '<div class="oneMessage1">\\n\' +\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=\'+value.courseId+\'"><span ><strong>\'+value.courseName+\'</strong></span></a>\\n\' +\n' +
                                           '                <span>发布考试《\'+value.examName+\'》，请进入该课程考试模块完成作业!</span>\\n\' +\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 4:
                                       htmStr = '<div class="oneMessage1" >\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>在名为《' + value.topicName + '》的帖子下发表的评论有新的回复，请查看!</span>\n' +
                                           '                <div>\n' +
                                           '\n' +
                                           '                    <div style="color:cornflowerblue">你的评论:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.commentInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '                <div>\n' +
                                           '\n' +
                                           '                    <div style="color:mediumpurple">他人回复:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.replyInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 5:
                                       htmStr = '<div class="oneMessage1">\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发表的名为《' + value.topicName + '》的帖子有新评论，请查看!</span>\n' +
                                           '                <div>\n' +
                                           '\n' +
                                           '                    <div style="color:cornflowerblue">评论详情:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.commentInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 6:
                                       htmStr = ' <div class="oneMessage1">\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的作业《' + value.testName + '》在' + value.testEndTime + '即将截止，请尽快进入该课程作业模块完成作业!</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 7:
                                       htmStr = ' <div class="oneMessage1">\n' +
                                           '                <a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的考试《' + value.examName + '》在' + value.examEndTime + '即将截止，请尽快进入该课程作业模块完成作业!</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 9:
                                       htmStr = '<div class="oneMessage1">\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的有关评论多次被举报违规，现已限制你的发言权，从<span style="color:rebeccapurple">' + value.forbidenBeginTime + '</span>起到<span style="color:rebeccapurple">' + value.forbidenEndTime + '</span>结束，你将无法进行发表评论活动</span>\n' +
                                           '                <div>\n' +
                                           '                    <div style="color:cornflowerblue">评论详情:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.commentInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 8:
                                       htmStr = '<div class="oneMessage1">\n' +
                                           '                你在<a href="courseIntroduce.jsp?courseid=' + value.courseId + '"><span ><strong>' + value.courseName + '</strong></span></a>\n' +
                                           '                <span>发布的有关帖子多次被举报违规，现已限制你的发帖权，从<span style="color:rebeccapurple">' + value.forbidenBeginTime + '</span>起到<span style="color:rebeccapurple">' + value.forbidenEndTime + '</span>结束，你将无法进行发表话题活动</span>\n' +
                                           '                <div>\n' +
                                           '                    <div style="color:cornflowerblue">帖子详情:</div>\n' +
                                           '                    <div style="margin-left:3%;">\n' + value.topicInfo +
                                           '                    </div>\n' +
                                           '                </div>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   case 10:
                                       htmStr = '<div class="oneMessage1">\n' +
                                           '                <span>' + value.proExcerciseTime + '发布实训《' + value.excerciseName + '》，请在综合实训一栏查看</span>\n' +
                                           '            </div>';
                                       $(".task").append(htmStr);
                                       break;
                                   default:
                                       htmStr = '';
                                       break;
                               }
                           }
                       })
                       if(isMessageFlush)
                       {
                           //send ajax to set messageState
                           $.ajax({
                               type: "POST",
                               url: "setMessageState?t=" + new Date().getTime(),
                               data: {"messageIdList":JSON.stringify(messageId)},
                               dataType:"json",
                               cache: false,
                               success: function (data) {
                                   $(".messageLabel").hide();
                               },
                               error: function (xhr) {
                                   alert(xhr.responseText);
                               }
                           })
                       }
                   },
                   error: function (xhr) {
                       alert(xhr.responseText);
                       $(".task").hide();
                       $(".nonetask").show();

                   }
               });
            $(".lefttasktitle").click(function () {
                $(".wait").hide();
                $(".finish").show();
                $(this).css("background", "green");
                $(".righttasktitle").css("background", "white");
            })
            $(".righttasktitle").click(function () {
                $(".finish").hide();
                $(".wait").show();
                $(this).css("background", "red");
                $(".lefttasktitle").css("background", "white");
            })
            $(".nav ").mouseover(function(e){
                var id=parseInt(e.target.id);
                switch(id){
                    case 1:$("#11").css("color","white");break;
                    case 3:$("#33").css("color","white");break;
                    case 4:$("#44").css("color","white");break;
                }
            })
            $(".nav").mouseout(function(){
                $(".nav a").css("color","grey");
            })
            $("#coursetype li").click(function(e){
                var id=parseInt(e.target.id);
                switch(id){
                    case 111:$("#111").css("color","orange");$("#222").css("color","grey");$("#333").css("color","grey");
                        $(".nonecourse").css("display","none");
                        $(".coursedetail").css("display","block");
                        $("#countofmaster").text("<%=courseCount%>");
                        break;
                    case 222:$("#222").css("color","orange");$("#111").css("color","grey");$("#333").css("color","grey");
                        var course="course";
                        var courseflag="courseflag";
                        var sb;
                        var coursecount=0;
                        for(i=1;i<=4;i++){
                            sb=courseflag+i;
                            if($("#"+sb).text()=="正在进行")
                            {
                                $("#"+course+i).css("display","block");
                                coursecount++;
                            }
                            else if($("#"+sb).text()=="已结束")
                                $("#"+course+i).css("display","none");
                        }
                        $("#countofmaster").text(coursecount);
                        if(coursecount!=0)
                        {
                            $(".nonecourse").css("display","none");
                        }
                        else {
                            $(".nonecourse").css("display","block");
                        }
                        break;
                    case 333:$("#333").css("color","orange");$("#222").css("color","grey");$("#111").css("color","grey");
                        //$(".nonecourse").css("display","block");
                        //$(".coursedetail").css("display","none");
                        var course="course";
                        var courseflag="courseflag";
                        var sb;
                        var coursecount=0;
                        for(i=1;i<=4;i++){
                            sb=courseflag+i;
                            if($("#"+sb).text()=="已结束")
                            {
                                $("#"+course+i).css("display","block");
                                coursecount++;
                            }
                            else if($("#"+sb).text()=="正在进行")
                                $("#"+course+i).css("display","none");
                        }
                        $("#countofmaster").text(coursecount);
                        if(coursecount!=0)
                        {
                            $(".nonecourse").css("display","none");
                        }
                        else {
                            $(".nonecourse").css("display","block");
                        }
                        break;
                }

            })

        })
        var deleteIds;
        var courseid = "";
        var sure;
        $(document).on("click", ".coursedelete", function () {
            sure= confirm("你确定退选本门课程吗？");
            if (sure) {
                deleteIds = $(this).attr("id");
                courseid = deleteIds.substr(12, 14);
                $.ajax({
                    type: "GET",
                    url: "studentDeleteCourse?t=" + new Date().getTime(),
                    data: { "deleteCourseId": courseid },
                    dataType: "json",
                    cache: false,
                    success: function (data) {
                        alert(data.result);
                        deleteIds = "";
                        courseid = "";
                        window.location.reload(true);
                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                    }
                })
                sure = "";
            }
            else {
                sure = "";
            }
        })
    </script>
</head>
<body onLoad="al('do')">
<div class="head">
    <div class="lefthead">
        <img src="./img/151946214719325.png" style="height:inherit">
    </div>
    <div class="righthead">
        <div class="navs">
            <div class="nav" id="1"><a id="11" href="./index.jsp">平台首页</a></div>
            <div class="nav" id="3"><a  id="33" href="#a">资讯</a></div>
            <div class="nav" id="4"><a  id="44" href="#a">帮助</a></div>
        </div>
    </div>
</div>
<div class="bgm">
    <div class="account">
        <div class="account1">
            <div class="leftaccount">
                <img  style=" float:left;height:180px;width:180px;border-radius:90px;margin-left:200px;" src="<%="./img/"+studentPic%>">
                <span style="float:left;margin-top:85px;width:auto;margin-left:10px;"><a style="color:white;font-size:25px;text-decoration: none;"><strong><%=studentName%></strong></a></span>
            </div>
            <div class="rightaccount">
                <div class="schoolinfo">
                    <div>来自:<span style="color:white;font-size:17px;"><%=belongSchName%></span></div>
                    <div >班级:<span style="color:white;"><%=belongClass%></span></div>
                </div>
                <div class="schoolinfo">
                    <div>论坛活跃度:<span style="color:navajowhite;font-size:17px;" id="activityValue"> </span>分</div>
                </div>
                <div style="float:left;margin-left:57px;margin-top:75px;">
                    <div id="alreadycourse"><strong><%=courseCount%></strong></div>
                    <div>已报名课程</div>
                </div>
                <div style="float:left;margin-left:7px;margin-top:75px;">
                    <div id="alreadycertify"><strong>0</strong></div>
                    <div>已获得证书</div>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="menu">
    <ul class="menus">
        <li style="margin-left:100px;"><div style="width:100px;height:inherit;font-size:17px;"><a href="students.jsp?do=course"><strong>我的课程</strong></a></div></li>
        <li><div style="width:100px;height:inherit;font-size:17px;"><a href="students.jsp?do=excercise"><strong>综合实训</strong></a></div></li>
        <li><div style="width:100px;height:inherit;font-size:17px;"><div style="float:left;"><a id="messageSpan" href="students.jsp?do=task" ><strong>我的消息</strong></a></div></div><div class="messageLabel" style="display:none;"><span id="messageNum" style="font-size: 7px;"></span></div></li>
        <li><div style="width:100px;height:inherit;font-size:17px;"><a href="students.jsp?do=score"><strong>成绩查询</strong></a></div></li>
        <li><div style="width:100px;height:inherit;font-size:17px;"><a href="students.jsp?do=capacity"><strong>能力档案</strong></a></div></li>
    </ul>
</div>
<div class="detail">
    <!--我的课程部分开始-->
    <div class="mycourseinfo">
        <div class="coursehead">
            <div style="float:left;width:10%;color:grey;font-size:19px;margin-top:20px;height:50%;margin-left:251px;">共<span id="countofmaster">4</span>门课程</div>
            <ul style="float:left;width:30%;margin-left:30%;color:grey;height:50%;margin-top:20px;" id="coursetype">
                <li id="111">全部</li>
                <li id="222">正在进行</li>
                <li id="333">已结束</li>
            </ul>
        </div>
        <div class="shiftdiv">
            <div class="nonecourse">
                暂无课程
            </div>
            <div class="coursedetail" id="course1">
                <div class="coursepic"><img src="./img/73bc9abfeaf7d04a8f0b72b3f2751f64.png" style="height:100%;border-style:none;">
                </div>
                <div class="rightofcoursepic">
                    <div class="coursename">
                        <h3 style="color:black"><strong>探索土星上的生命</strong></h3>
                        <span style="background-color:rgba(0,0,0,0.4);color:white;fon-size:17px;" id="courseflag1">正在进行</span>
                    </div>
                    <div class="coursedisplay">
                        <div class="coursebar" id="coursebar1"></div>
                        <div class="courseprogress"><span>0</span>/<span>6</span></div>
                        <div class="courseDirectoryEntry"><a href="#a">进入学习</a></div>
                        <div class="coursedelete">取消课程</div>
                    </div>
                </div>
            </div>
            <div class="coursedetail" id="course2">
                <div class="coursepic"><img src="./img/c2ceef388020ab9f29c829b1b9a218c4.jpg" style="height:100%;border-style:none;">
                </div>
                <div class="rightofcoursepic">
                    <div class="coursename">
                        <h3 style="color:black;"><strong>M78星云探索发现</strong></h3>
                        <span style="background-color:rgba(0,0,0,0.4);color:white;fon-size:17px;" id="courseflag2">正在进行</span>
                    </div>
                    <div class="coursedisplay">
                        <div class="coursebar" id="coursebar2"></div>
                        <div class="courseprogress"><span>0</span>/<span>13</span></div>
                        <div class="courseDirectoryEntry"><a href="#a">进入学习</a></div>
                        <div class="coursedelete">取消课程</div>
                    </div>
                </div>
            </div>
            <div class="coursedetail" id="course3">
                <div class="coursepic"><img src="./img/c2ceef388020ab9f29c829b1b9a218c4.jpg" style="height:100%;border-style:none;">
                </div>
                <div class="rightofcoursepic">
                    <div class="coursename">
                        <h3 style="color:black;"><strong>计算机组成原理</strong></h3>
                        <span style="background-color:rgba(0,0,0,0.4);color:white;fon-size:17px;" id="courseflag3">已结束</span>
                    </div>
                    <div class="coursedisplay">
                        <div class="coursebar" id="coursebar5"></div>
                        <div class="courseprogress"><span>0</span>/<span>8</span></div>
                        <div class="courseDirectoryEntry"><a href="#a">进入学习</a></div>
                        <div class="coursedelete">取消课程</div>
                    </div>
                </div>
            </div>
            <div class="coursedetail" id="course4">
                <div class="coursepic"><img src="./img/1425026230.jpg" style="height:100%;border-style:none;">
                </div>
                <div class="rightofcoursepic">
                    <div class="coursename">
                        <h3 style="color:black;"><strong>数据结构</strong></h3>
                        <span style="background-color:rgba(0,0,0,0.4);color:white;fon-size:17px;" id="courseflag4">已结束</span>
                    </div>
                    <div class="coursedisplay">
                        <div class="coursebar" id="coursebar3"></div>
                        <div class="courseprogress"><span>0</span>/<span>9</span></div>
                        <div class="courseDirectoryEntry"><a href="#a">进入学习</a></div>
                        <div class="coursedelete">取消课程</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="shift">helloawefi</div>
    </div>

    <!--我的课程结束-->

    <!--综合实训开始-->
    <div class="myexcerciseinfo">

    </div>
    <!--综合实训结束-->

    <!--我的任务开始-->
    <div class="mytaskinfo">
        <div class="nonetask"><h2><strong style="color:black;">暂无任务</strong></h2></div>
        <div class="task">
            <!--
                消息包括：
                1.课程开课进度
                2.提醒课程作业发布
                3.课程考试发布
                4.评论回复提醒
                5.考试作业截止提醒
                6.实训发布提醒
                7.评论禁言提醒
            <div class="oneMessage">
                <a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>开课了，欢迎进入该课程学习!</span>
            </div>
            <hr>
            <div class="oneMessage">
                <a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发布作业《第一单元测试》，请进入该课程作业模块完成作业!</span>
            </div>
            <hr>
            <div class="oneMessage">
                <a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发布考试《期末考试》，请进入该课程考试模块完成考试!</span>
            </div>
            <hr>
            <div class="oneMessage" >
                你在<a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>在名为《如何学习C语言》的帖子下发表的评论有新的回复，请查看!</span>
                <div>

                    <div style="color:cornflowerblue">你的评论:</div>
                    <div style="margin-left:3%;">
                        hehe
                    </div>
                </div>
                <div>

                    <div style="color:mediumpurple">他人回复:</div>
                    <div style="margin-left:3%;">
                        aoeirgoeirgeriouerutoraowiegirureoiug
                    </div>
                </div>
            </div>
            <hr>
            <div class="oneMessage">
                你在<a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发表的名为《如何学习C语言》的帖子有新评论，请查看!</span>

            </div>
            <hr>
            <div class="oneMessage">
                你在<a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发表的名为《如何学习C语言》的帖子有新评论，请查看!</span>
                <div>

                    <div style="color:cornflowerblue">评论详情:</div>
                    <div style="margin-left:3%;">
                    </div>
                </div>
            </div>
            <hr>
            <div class="oneMessage">
                <a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发布的作业《第二单元测验》在5月3日即将截止，请尽快进入该课程作业模块完成作业!</span>
            </div>
            <hr>
            <div class="oneMessage">
                <a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发布的考试《期末考试》于5月3日即将截止，请尽快进入该课程考试模块完成考试!</span>
            </div>
            <div class="oneMessage">
                <a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发布的考试《期末考试》于5月3日即将截止，请尽快进入该课程考试模块完成考试!</span>
            </div>
            <div class="oneMessage">
                你在<a href="courseIntroduce.jsp?courseid=1"><span ><strong>计算机组成原理</strong></span></a>
                <span>发布的有关评论多次被举报违规，现已限制你的发言权，从<span style="color:rebeccapurple">2019年3月23日</span>起到<span style="color:rebeccapurple">2019年5月6日</span>结束，你将无法进行发表评论活动</span>
                <div>
                    <div style="color:cornflowerblue">评论详情:</div>
                    <div style="margin-left:3%;">
                    </div>
                </div>
            </div>
            <hr>
        </div>-->
        </div>

    </div>
    <!--我的任务结束-->

    <!--我的成绩开始-->
    <div class="myscoreinfo">
        <div class="nonescore"><h1 style="color:black;"><strong>暂时还没有成绩</strong></h1><span style="color:grey;font-size:15px;"><strong>在任课老师改完习题后，您会看到自己的学习成果~</strong></span></div>
        <div class="scorehead">
            <div style="float:left;width:10%;color:grey;font-size:19px;margin-top:20px;height:50%;margin-left:251px;">共<span id="countofscore">3</span>门成绩</div>
            <ul style="float:left;width:30%;margin-left:30%;color:grey;height:50%;margin-top:20px;" id="puttime">
                <li id="444">一天内</li>
                <li id="555">一星期内</li>
                <li id="666">半年内</li>
                <li id="777">所有成绩</li>
            </ul>
        </div>
        <div class="listofscore">
            <table style="width:90%;text-align: center;" id="ScoreTable" >
                <tr id="tabletitle">
                    <td style="color:black;"><strong style="font-size:16px">所属课程</strong></td>
                    <td style="color:black;"><strong style="font-size:16px">成绩名称</strong></td>
                    <td style="color:black;"><strong style="font-size:16px">成绩类型</strong></td>
                    <td style="color:black;"><strong style="font-size:16px">分数</strong></td>
                </tr>
                <!--<tr>
                    <td id="belongCourse1" >无</td>
                    <td id="scorename1" >android五子棋</td>
                    <td id="teachername1" >综合实训</td>
                    <td id="score1" style="color:green;">86</td>
                </tr>
                <tr>
                    <td id="belongCourse2" ><a href="#">探索土星上的生命 </a> </td>
                    <td id="scorename2">第一单元测试</td>
                    <td id="teachername2">测试</td>
                    <td id="score2" style="color:red;">59</td>
                </tr>
                <tr>
                    <td id="belongCourse3" ><a href="#">数据结构 </a></td>
                    <td id="scorename3">数据结构期末考试</td>
                    <td id="teachername3">考试</td>
                    <td id="score3" style="color:green;">100</td>
                </tr>-->
            </table>
        </div>
    </div>
    <!--我的成绩结束-->

    <!--我的能力开始-->
    <div class="mycapacityinfo">
        <div class="noneinfo"><h2><strong>暂无能力档案</strong></h2><span><strong style="color:grey;">亲，要继续加油，多参加一些活动项目拿证书才会有能力档案~</strong></span></div>
        <div class="capacityhead">

        </div>
        <div class="capacitydetail">
            <h3 ><strong style="color:black;">能力档案详情</strong></h3>
            <table class="table">
                <thead>
                <tr>
                    <td>课程名称</td>
                    <td>论坛分数（100）</td>
                    <td>课后作业分数（100）</td>
                    <td>期末测试分数（100）</td>
                    <td>总分（100）</td>
                    <td>证书类型</td>
                </tr>
                </thead>
                <tbody id="capacityTBody">
                </tbody>
            </table>
        </div>

    </div>
    <!--我的能力结束-->
</div>
</body>
</html>


