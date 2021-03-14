<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/4/1 0001
  Time: 8:29
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.xqq.pojo.Teacher" %>
<%@ page import="com.xqq.pojo.Student" %>
<% Teacher teacher=(Teacher)session.getAttribute("teacherInfo");
    Student student=(Student)session.getAttribute("studentInfo");
    Integer commentWritterId=-1;
    Integer commentWritterType=-1;
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
        commentWritterId=teacherId;
        commentWritterType=1;
    }
    else if(student!=null)
    {
        // System.out.println("??student !=null");
        studentId=student.getStudentId();
        studentName=student.getStudentName();
        studentPic=student.getStudentPic();
        commentWritterId=studentId;
        commentWritterType=0;
    }
    //System.out.println("nullStudent");
%>
<html >
<head >
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>学习页面</title>
</head>
<link href="css/study.css" rel="stylesheet" type="text/css">
<link rel="stylesheet" href="./css/bootstrap.min.css">
<script src="./js/jquery-2.0.3.js"></script>
<script type="text/javascript" src="./js/bootstrap.min.js"></script>
<link href="./css/video-js.css" rel="stylesheet" type="text/css">
<script src="./js/video.js"></script>
<script>
    var finishnoticehtml = "";
    var emotionType=["鄙视","滑稽","苦笑","微笑","开心","发怒","悲伤","笑哭","白眼","大哭","调皮","爱心","恐慌"];
    var commentDetail="";
    var hasEmotion=false;
    var lastLocation=0;//评论内容截取的起点
    function hh(str) {
        var reg = new RegExp("<br/>", "g");
        str = str.replace(reg, "\n");
        return str;
    }
    function changeToPath(str){//将部分字符串转换为文件路径中的字符串
        if(str=="鄙视")
            return "img/bishi.jpg";
        else if(str=="调皮")
            return "img/tiaopi.jpg";
        else if(str=="微笑")
            return "img/weixiao.jpg";
        else if(str=="大哭")
            return "img/daku.jpg";
        else if(str=="滑稽")
            return "img/huaji.jpg";
        else if(str=="悲伤")
            return "img/beishang.jpg";
        else if(str=="发怒")
            return "img/fanu.jpg";
        else if(str=="笑哭")
            return "img/xiaoku.jpg";
        else if(str=="开心")
            return "img/kaixin.jpg";
        else if(str=="恐慌")
            return "img/konghuang.png";
        else if(str=="苦笑")
            return "img/kuxiao.jpeg";
        else if(str=="白眼")
            return "img/baiyan.jpg";
        else if(str=="爱心")
            return "img/aixin.jpg";
        else return "";
    }
    function changeToLabel(str){//将部分字符串转换为文件路径中的字符串
        if(str=="img/bishi.jpg")
            return "鄙视";
        else if(str=="img/tiaopi.jpg")
            return "调皮";
        else if(str=="img/weixiao.jpg")
            return "微笑";
        else if(str=="img/daku.jpg")
            return "大哭";
        else if(str=="img/huaji.jpg")
            return "滑稽";
        else if(str=="img/beishang.jpg")
            return "悲伤";
        else if(str=="img/fanu.jpg")
            return "发怒";
        else if(str=="img/xiaoku.jpg")
            return "笑哭";
        else if(str=="img/kaixin.jpg")
            return "开心";
        else if(str=="img/kuxiao.jpeg")
            return "苦笑";
        else if(str=="img/baiyan.jpg")
            return "白眼";
        else if(str=="img/aixin.jpg")
            return "爱心";
        else return "";
    }
    function isEmo(str){//判断[]里的是否为表情
        var isEmotion=false;
        for(var i=0;i<emotionType.length;i++){
            if(emotionType[i]==str)
            {
                isEmotion=true;
                break;
            }
        }
        return isEmotion;
    }
    function changeToHtmlStr(str){
        var tempStr="";
        var htmlStr="";//最终转换的html字符串
       // alert(str);
        for(var i=0;i<str.length;i++)
        {
            var flag=false;
            if(str[i]!='[')
            htmlStr+=str[i].toString();
            if(str[i]=='['){//如果是左括号 开始检测是否为某个表情
                var j=i+1;
                while(str[j]!=']'&&j<str.length)//当还没到 ] 并且还没结束时循环
                {
                    if(str[j]!='[')
                    tempStr+=str[j++].toString();
                    else
                    {
                        htmlStr+=str[i].toString()+tempStr;
                        i=j-1;
                        tempStr="";
                        flag=true;
                        break;
                    }
                }
                if(!flag) {//当上面没有发生时
                    if (str[j] == ']') {
                       // alert("in [] is:" + tempStr);
                        if (isEmo(tempStr))//如果是表情 ，则替换
                        {
                            var pathStr = changeToPath(tempStr);
                            htmlStr += '<img style="width:23px;height:23px;" src="' + pathStr + '">';
                            i = j;
                        }
                        else {
                            htmlStr += '['.toString() + tempStr + ']'.toString();
                            i = j;
                        }
                    }
                    else {
                        htmlStr += '['.toString() + tempStr;
                        i = j;
                    }
                }
                tempStr="";
            }
        }
        //alert("htmlStr:"+htmlStr);
        return htmlStr;
    }
    function isImgCode(str){
        var imgCodeStr='img';
        for(var i=0;i<str.length&&i<imgCodeStr.length;i++){
            if(imgCodeStr[i]==str[i]){
                continue;
            }else break;
        }
        //此时说明是中断，说明不是图片
        if(i<str.length&&i<imgCodeStr.length){
            return "";
        }else if(i<imgCodeStr.length){
            return "";
        }
        //说明此时比较完imgCodeStr,str仍然有剩余，则说明是img标签
        else {
            var htmlStr='<'+str+'>';
            $("#imgStr").empty();
            $("#imgStr").append(htmlStr);
            var src=$("#imgStr img").attr("src");
            var label=changeToLabel(src);
            return label;
        }
    }

    function changeToDBstr(str){
        var tempStr="";
        var htmlStr="";
        for(var i=0;i<str.length;i++)
        {
            var flag=false;
            if(str[i]!='<')
                htmlStr+=str[i].toString();
            if(str[i]=='<'){//如果是左括号 开始检测是否为某个表情
                var j=i+1;
                while(str[j]!='>'&&j<str.length)//当还没到 > 并且还没结束时循环
                {
                    if(str[j]!='<')
                        tempStr+=str[j++].toString();
                    else
                    {
                        htmlStr+=str[i].toString()+tempStr;
                        i=j-1;
                        tempStr="";
                        flag=true;
                        break;
                    }
                }
                if(!flag) {//当上面没有发生时
                    if (str[j] == '>') {
                         //alert("in <> is:" + tempStr);
                         var label=isImgCode(tempStr);
                        if (label!="")//如果是表情图片代码字符串 ，则替换
                        {
                            htmlStr += '['+label+']';
                            i = j;
                        }
                        else {
                            htmlStr += '<'.toString() + tempStr + '>'.toString();
                            i = j;
                        }
                    }
                    else {
                        htmlStr += '<'.toString() + tempStr;
                        i = j;
                    }
                }
                tempStr="";
            }
        }
        return htmlStr;
    }
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
    function submitForm(){
        $("#examPramOfCourseId").val(courseids)
        if(hasSubmit==true)
            return false;
        hasSubmit=true;
        return true;
    }
    var arrys = [];//记录每道大题中小题的数目
    var  arryTypes = [];//记录每道大题的类型
    var trueAnswer = [];//记录每一道题目的正确答案
    var score = [];//记录每道大题中每小题的分值
    var courseids = "";
    var topicArry=[];
    $(document).ready(function () {
        var teststr = "";
        var chapterArr = [];
        courseids = getUrlParam("courseid");
        $("#courseIds").val(courseids);
        $("#afterLogin").show();
        var studentPic = "./img/" + "<%=studentPic%>";
        $("#Pic").attr("src", studentPic);
        $("#afterLogin").attr("title", "<%=studentName%>");
        $("#hreftointro").attr("href", "courseintroduce.jsp?courseid=" + courseids);
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
        $.ajax({
            type: "GET",
            url: "getCourse?time=" + new Date().getTime(),
            cache: false,
            data: { "courseid": courseids },
            dataType: "json",
            success: function (data) {
                $("title").text(data.courseName + "-学习页面");
                $("#coursePostSrc").attr("src", "./img/" + data.courseImgSrc);
                $("#coursename").text(data.courseName);
                $("#welcomeName").text("欢迎来到《" + data.courseName + "》");
                $("#schoolSrc").attr("src", "./img/" + data.comPic);
            },
            error: function (xhr) {
                alert(xhr.responseText);
            }
        })
        $.ajax({
            type: "GET",
            url: "noticeinfo?time=" + new Date().getTime(),
            cache: false,
            data:{"courseid":courseids},
            dataType: "json",
            success: function (data) {
                $(".noticedetail").empty();
                $.each(data, function (i, values) {
                    if (values["noticeState"] == "finish") {
                        finishnoticehtml += '<div class="onenotice">' +
                            "<h3>" + values["noticeTitle"] + "</h3>" +
                            '<textarea style="color:darkgrey;border-style:none;background: rgba(248,222,222,0.1);" cols="120" rows="14" readonly>' +
                            hh(values["noticeBody"]) + "</textarea>" +
                            '<p style="text-align:right;width:100%;heigth:auto;">' + values["noticeWritter"] + values["noticeTime"] + "</p>" +
                            '<span style="color:rgba(231,227,227,0.5);background:rgba(0,0,0,0.4);width:30px;margin-left:80%;height:auto;border-radius: 2px;">' + values["noticeSystemTime"] + "</span></div>";
                    }
                })
                $(".noticedetail").html(finishnoticehtml);
                finishnoticehtml = "";
            },
            error: function (xhr) {
                alert(xhr.responseText);
                $(".nonenotice").show();
            }

        });
        $("#lists div").eq($("#notice").index()).removeClass("whites");
        $("#lists div").eq($("#notice").index()).addClass("greens").siblings().removeClass("greens");
        $("#lists div").eq($("#notice").index()).siblings().addClass("whites");
        $(".noticedisplay").show();
        $(".evaluationdisplay").hide();
        $(".coursewaredisplay").hide();
        $(".testdisplay").hide();
        $(".examdisplay").hide();
        $(".discussdisplay").hide();
        var videojsonarray = [];
        var textjsonarray = [];
        var quaters = new Array();
        var bigTestTypeNum = 0;
        $.ajax({
            type: "GET",
            url: "chapterinfo?t=" + new Date().getTime(),
            data:{"courseid":courseids},
            cache: false,
            dataType: "json",
            success: function (data) {
                $(".coursewaredetail").empty();
                $("#chapter").empty();
                $(".categories").empty();
                $.each(data, function (i, values) {
                    var sbs = [];
                    var cou = i + 1;
                    var strhtml1 = "";
                    var strhtml3="";
                    strhtml1 = '<div class="courseutil" id="' + values["chapterId"]+ '">' +
                        '<div class="utiltitle" id="utiltitle' + values["chapterId"] + '">' +
                        '<img src="./img/t01501c515ccc7294fe.png" style="height:15px;margin-left:5px;" id="logo' + values["chapterId"] + '">&nbsp' + values["titleNumberName"] + '<span style="margin-left:14px;">' + values["titleName"] + "</span>" +
                        "</div>" +
                        '<div class="utildetails" id="utildetails' + values["chapterId"] + '">' +
                        "</div>" + "</div>";
                    strhtml3='<div class="group" id="group'+values.chapterId+'">\n' +
                        '<div class="categorytitle">'+values.titleNumberName+"&nbsp"+ values.titleName+ "</div>\n" +
                        '<div class="categorydetails" id="categorydetails'+values.chapterId+'">\n' +
                        "</div>\n"+
                        "</div>";
                    $(".categories").append(strhtml3);
                    $(".coursewaredetail").append(strhtml1);
                    $.ajax({
                        type: "GET",
                        url: "utilinfo?t=" + new Date().getTime(),
                        data:{"chapterId":values["chapterId"]},
                        cache: false,
                        dataType: "json",
                        success: function (data) {
                            var str="";
                            var strhtml2="";
                            $.each(data,function(j,vals){
                                strhtml2='<div class="utildetail" id="'+vals.utilId+'">'+vals.utilDetailTitle+"&nbsp"+vals.utilDetailName+
                                '<span style="margin-left:50px;color:greenyellow;font-size:15px;">共'+vals.utilDetailVideoNumber+"个视频</span>"+
                                '<span style="margin-left:20px;color:greenyellow;font-size:15px;">'+vals.utilDetailTextNumber+"份文档</span>"+
                                "</div>";
                                str= '<div class="categorydetail" id="util'+vals.utilId+'">'+vals.utilDetailTitle +vals.utilDetailName+"</div>\n";
                                $("#categorydetails" + values["chapterId"]).append(str);
                                $("#utildetails" + values["chapterId"]).append(strhtml2);
                            })
                        },
                        error: function (xhr) {
                            alert(xhr.responseText);
                        }
                    })
                })
            },
            error: function (xhr) {
                alert(xhr.responseText);
                alert("fail");
            }
        })
        $(document).on("click", ".utiltitle", function () {
            var id = "#" + $(this).attr("id");
            var num = id.substring(10, 12);
            var id1 = "#logo" + num;
            var id2 = "#utildetails" + num;
            if ($(id2).is(":hidden")) {
                $(id2).show();
                $(id1).attr("src", "./img/t0179c6050a271622e0.jpg");
            }
            else {
                $(id2).hide();
                $(id1).attr("src", "./img/t01501c515ccc7294fe.png");
            }
        })
        $(document).on("click", ".utildetail", function () {
            $(".videos").empty();
            var idr = $(this).attr("id");
            $(".categorydetail").css("background-color", "white");
            $("#util"+idr).css("background-color","yellow");
            var strh = "";
            $.ajax({
                type: "GET",
                url: "getSource?t=" + new Date().getTime(),
                data: { "utilId": idr },
                cache: false,
                dataType: "json",
                success: function (data) {
                    $(".videos").empty();
                    $(".textDetail").empty();
                    $.each(data, function (i, values) {
                        if(values.sourceType=="video") {
                            strh = '<div class="avideo">' +
                                '<video controls preload="auto" width="528px" height="300px"  poster="' + "./img/" + values.videoposter + '"' +
                                ' data-setup="{}">' + '<source src="' + "./video/" + values.sourceSrc + '" type="video/mp4" />' + "</video></div>";
                             //alert(strh);
                            $(".videos").append(strh);
                        }
                        else
                        {
                            strh = " <div >\n" +
                                '<a  href="./Texts/'+values.sourceSrc+'" download="'+values.sourceSrc+'">下载课件'+values.sourceSrc+"</a>\n" +
                                "</div>";
                            // alert(strh);
                            $(".textDetail").append(strh);
                        }
                    });
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }

            })
            $(".studyWare").show();
            $(".coursewaredetail").hide();
            $(".videoview").show();
            $("#gobackutil").show();
        })
        $(document).on("click", ".categorydetail", function () {
            $(".videos").empty();
            var idr = $(this).attr("id").substr(4,6);
            $(".categorydetail").css("background-color","white");
            $(this).css("background-color","yellow");
            //alert(idr);
            var strh = "";
            $.ajax({
                type: "GET",
                url: "getSource?t=" + new Date().getTime(),
                data: { "utilId": idr },
                cache: false,
                dataType: "json",
                success: function (data) {
                    $(".videos").empty();
                    $(".textDetail").empty();
                    $.each(data, function (i, values) {
                        if(values.sourceType=="video") {
                            strh = '<div class="avideo">' +
                                '<video controls preload="auto" width="528px" height="300px"  poster="' + "./img/" + values.videoposter + '"' +
                                ' data-setup="{}">' + '<source src="' + "./video/" + values.sourceSrc + '" type="video/mp4" />' + "</video></div>";
                            //alert(strh);
                            $(".videos").append(strh);
                        }
                        else
                        {
                            strh = " <div >\n" +
                                '<a  href="./Texts/'+values.sourceSrc+'" download="'+values.sourceSrc+'">下载课件'+values.sourceSrc+"</a>\n" +
                                "</div>";
                           // alert(strh);
                            $(".textDetail").append(strh);
                        }
                    });
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }

            })
            //$(".videos").show();
            //$(".coursewaredetail").hide();
           // $(".videoview").show();
            //$("#gobackutil").show();
        })

        $("#lists div").click(function () {
            $("#lists div").eq($(this).index()).removeClass("whites");
            $("#lists div").eq($(this).index()).addClass("greens").siblings().removeClass("greens");
            $("#lists div").eq($(this).index()).siblings().addClass("whites");
            if ($("#lists div").eq($(this).index()).text() == "公告") {
                $(".noticedisplay").show();
                $(".evaluationdisplay").hide();
                $(".coursewaredisplay").hide();
                $(".testdisplay").hide();
                $(".examdisplay").hide();
                $(".discussdisplay").hide();
            }
            else if ($("#lists div").eq($(this).index()).text() == "评分标准") {
                $.ajax({
                    type: "GET",
                    url: "evaluationinfo",
                    cache: false,
                    data:{"courseid":courseids},
                    dataType: "json",
                    success: function (data) {
                        var evaluationdetail = hh(data["evaluationinfo"]);
                        $("#evaluationdetail").val(evaluationdetail);
                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                        alert("fail");
                    }
                })
                $(".noticedisplay").hide();
                $(".evaluationdisplay").show();
                $(".coursewaredisplay").hide();
                $(".testdisplay").hide();
                $(".examdisplay").hide();
                $(".discussdisplay").hide();
            }
            else if ($("#lists div").eq($(this).index()).text() == "课件") {
                $(".noticedisplay").hide();
                $(".evaluationdisplay").hide();
                $(".coursewaredisplay").show();
                $(".testdisplay").hide();
                $(".examdisplay").hide();
                $(".discussdisplay").hide();
            }
            else if ($("#lists div").eq($(this).index()).text() == "作业") {
                chapterArr = [];
                $.ajax({
                    type: "GET",
                    url: "getTest?t=" + new Date().getTime(),
                    data: { "courseId": courseids },
                    dataType: "json",
                    cache: false,
                    success: function (data) {
                        //var strhtm='<div class="tests">'+
                        //	'<div class="testtitle">'+data.chapterName+"</div>"+
                        //"</div>";
                        $.each(data, function (i, val) {
                            var flag = false;
                            for (var m = 0; m < chapterArr.length; m++) {//将获取的json数组按每一章分开存储
                                if (chapterArr[m][0].chapterId == val.chapterId) {
                                    chapterArr[m].push(val);
                                    flag = true;
                                    break;
                                }
                            }
                            if (flag == false) {
                                var arry = [];
                                arry.push(val);
                                chapterArr.push(arry);
                            }

                        })
                        $(".testrough").empty();
                        $.each(chapterArr, function (k, values) {//values是每一章对应的作业，有多个作业
                            var strhtm='<div class="tests" id="tests'+values[0].chapterId+'">'+
                                '<div class="testtitle">'+values[0].chapterName+"</div>"+
                                "</div>";
                            $(".testrough").append(strhtm);
                            $("#tests"+values[0].chapterId).empty();
                            $.each(values, function (n, v) {//每一章里的每一个作业
                                var strh = '<div class="tests">'+
                                    '<div class="testtitle">'+v.chapterName+"</div>"+
                                    '<div class="testdetails">'+
                                    '<div class="testdetailtitle">'+'<div style="width:30%;height:auto;float:left;margin-top:7px;">'+v.testName+'<span style="color:white;background:rgba(151,146,146,0.5);font-size:15px;display:inline;">截止时间:'+v.testEndTime;
                                    if(v.testState!="已截止")
                                        strh+='</span></div><input type="button" value="查看" class="gotofinish" id="'+v.testId+'"></div>';
                                    else
                                        strh+='</span></div><input type="button" value="查看" style="background-color: darkgray;" class="gotofinish" id="'+v.testId+'" disabled></div>';
                                    strh+='<div class="testdetail">'+
                                    '<span>发布时间:'+v.testStartTime+"</span>"+
                                    "<span>截止时间:"+v.testEndTime+"</span>"+
                                    '<span>总分合计:'+v.testAllMark+"分</span>"+
                                    '<div><span style="float:left;">有效提交次数：</span>'+'<span id="submitCount'+v.testId+'" style="float:left;" >'+v.submitCount+"</span>"+'<span style="float:left;">次,取最高分数为有效成绩</span></div>'+
                                    '&nbsp&nbsp&nbsp<span style="color:orange;">状态:'+v.testState+"</span>"+
                                    "</div>";
                                $("#tests" + values[0].chapterId).append(strh);
                                //alert(v.testDetail);
                            })

                        })
                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                    }
                })
                $(".noticedisplay").hide();
                $(".evaluationdisplay").hide();
                $(".coursewaredisplay").hide();
                $(".testdisplay").show();
                $(".examdisplay").hide();
                $(".discussdisplay").hide();
            }
            else if ($("#lists div").eq($(this).index()).text() == "考试") {

                $(".noticedisplay").hide();
                $(".evaluationdisplay").hide();
                $(".coursewaredisplay").hide();
                $(".testdisplay").hide();
                $(".examdisplay").show();
                $(".discussdisplay").hide();
                $.ajax({
                    type: "GET",
                    url: "getExam?t=" + new Date().getTime(),
                    data:
                        {
                            "courseid": courseids
                        },
                    cache: false,
                    dataType: "json",
                    success:function(data){
                        if(data.examPass==1){
                            $("#examTitle").html($("#coursename").val()+"期末测试");
                            $("#examSuplement").html(data.examSuplement);
                            $("#startTime").html(data.examStartTime);
                            $("#endTime").html(data.examEndTime);
                            $(".ExamState").html("状态:"+data.isFinish+","+data.isEnd);
                            $("#limitTime").html(data.examLimitTime);
                            $("#oneProblemTitle").html(data.examOneTitle);
                            $("#twoProblemTitle").html(data.examTwoTitle);
                            $("#threeProblemTitle").html(data.examThreeTitle);
                            $("#fourProblemTitle").html(data.examFourTitle);
                            $("#fiveProblemTitle").html(data.examFiveTitle);
                            if(data.isEnd=="end"){
                                $("#beginExam").attr("disabled",true);
                            }else{
                                //如果已完成，则只能查看
                                if(data.isFinish=="finish"){
                                    $("#submitExamBtn").attr("disabled",true);
                                    $(".chooseFile").attr("disabled",true);
                                    var strhtmOne='<img style="width:65%;height:inherit;margin-left:20%;" src="./img/'+data.oneSrc+'"/>';
                                    $("#problemAnswerSrcLookThroughOne").html(strhtmOne);
                                    var strhtmTwo='<img style="width:65%;height:inherit;margin-left:20%;" src="./img/'+data.twoSrc+'"/>';
                                    $("#problemAnswerSrcLookThroughTwo").html(strhtmTwo);
                                    var strhtmThree='<img style="width:65%;height:inherit;margin-left:20%;" src="./img/'+data.threeSrc+'"/>';
                                    $("#problemAnswerSrcLookThroughThree").html(strhtmThree);
                                    var strhtmFour='<img style="width:65%;height:inherit;margin-left:20%;" src="./img/'+data.fourSrc+'"/>';
                                    $("#problemAnswerSrcLookThroughFour").html(strhtmFour);
                                    var strhtmFive='<img style="width:65%;height:inherit;margin-left:20%;" src="./img/'+data.fiveSrc+'"/>';
                                    $("#problemAnswerSrcLookThroughFive").html(strhtmFive);
                                    $(".HotDate").hide();
                                }else if(data.isFinish=="notFinish"){
                                    //如果未完成，则开启计时
                                        limitTime=data.examLimitTime;
                                        interval= setInterval(getRTime,1000);
                                }
                            }

                        }else{
                            $(".examdisplay").empty();
                        }
                    },
                    error:function(e){
                        alert(e.responseText);
                    }
                })
            }
            else if ($("#lists div").eq($(this).index()).text() == "讨论区") {
                $(".noticedisplay").hide();
                $(".evaluationdisplay").hide();
                $(".coursewaredisplay").hide();
                $(".testdisplay").hide();
                $(".examdisplay").hide();
                $(".discussrough").empty();
                $.ajax({
                    type: "GET",
                    url: "showTopic?t=" + new Date().getTime(),
                    data: { "courseId": courseids },
                    dataType: "json",
                    cache: false,
                    success: function (data) {
                        topicArry=data;
                        var strhtm="";
                        $.each(data, function (i, val) {
                            strhtm+=" <div class=\"adiscussrough\">\n" +
                                "                <div><span>"+val.topicDetail+"</span></div>\n" +
                                '                <div style="width:100%;height:auto;"><div style="float:left;width:35%;"><span id="writer'+val.topicWriterId+'" style=\"color:green;font-size:15px;display:inline;\">'+val.topicWriterName+'</span><span style=\"color:darkgrey;display:inline;font-size:15px;margin-left:15px;\">于'+val.topicTime+"发表</span>\n" +
                                '\t\t\t\t<span style=\"color:green;display:inline;font-size:14px;margin-left:15px;\" id="topic'+val.topicId+'"class="topicDetail">详情</span></div>\n' +
                                '\t\t\t\t<div style=\"float:left;width:64%;\"><span style=\"color:darkgrey;display:inline;font-size:14px;margin-left:470px;\">浏览:'+val.readCount+"</span>\n" +
                                "\t\t\t\t<span style=\"color:darkgrey;display:inline;font-size:14px;margin-left:15px;\">评论:"+val.comNum+"</span>\n" +
                                "\t\t\t\t<span style=\"color:darkgrey;display:inline;font-size:14px;margin-left:15px;\">举报("+val.reportNum+")</span></div></div>\n" +
                                "            </div>";
                            if(i==0)
                                console.log(strhtm);
                        })
                        $(".discussrough").html(strhtm);
                    },
                    error: function (xhr) {
                        alert(xhr.responseText);
                    }
                })
                //alert("discussdisplay.show()");
                $(".discussdisplay").show();
            }
        })

        $("#gobackutil").click(function () {
            $(this).hide();
            $(".videoview").hide();
           // $(".textDetail").hide();
            $(".textDetail").empty();
            $(".videos").empty();
            $(".coursewaredetail").show();
        })
        $(".utildetail").click(function () {
            $(".coursewaredetail").hide();
            $(".videoview").show();
            $("#gobackutil").show();
           // $(".textDetail").show();
        })
        $("#undertaking").change(function () {
            if ($("#undertaking").is(":checked")) {
                $("#begintests").css("background", "rgba(131,207,32,1.00)");
                $("#begintests").removeAttr("disabled", "disabled");
            }
            else {
                $("#begintests").css("background", "darkgrey");
                $("#begintests").attr("disabled", "disabled");
            }
        })
        $("#beginExam").click(function () {
            if(firstClick) {
                var NowTime = new Date();
                var EndTimes = NowTime.getTime() + limitTime * 60 * 1000;
                EndTime = new Date(EndTimes);
                firstClick=false;
            }
            $("#examEntry").hide();
            $("#examDetail").show();
        })
        $("#gobackrough").click(function () {
            $(".testentry").hide();
            $(".testrough").show();
        })
        $("#gobackentry").click(function () {
            $(".testread").hide();
            $(".testentry").show();
            $("#gobackrough").show();
        })
        $(document).on("click", ".gotofinish", function () {//查看 按钮
            arrys = [];//记录每道大题中小题数目
            teststr = "";
            arryTypes = [];
            trueAnswer = [];
            score = [];
            $("#theGetMarks").empty();
            var id1 = $(this).attr("id");
            $("#testIds").val(id1);
            var sb="#testIds";
            var id2="#submitCount"+$(sb).val();
            var submitCount=$(id2).html();
            //alert("MaxCount:"+submitCount);
            $.ajax({
                type: "GET",
                url: "getScoreAndSubmitCount?t=" + new Date().getTime(),
                data:
                    {
                        "studentId": $("#studentIds").val(),
                        "testId": $("#testIds").val(),
                    },
                cache: false,
                dataType: "json",
                success:function(data){
                    if(data.submitCount!=-1)//此时表明该学生提交过这个作业
                    {
                        var sb="#testIds";
                        var id1="#submitCount"+$(sb).val();
                        var submitCount=$(id1).html();
                        var submitC=submitCount-data.submitCount;
                        if(submitC<=0)//如果提交次数达到上限，则显示成绩，重做按钮和提交按钮都不能按
                        {
                            $("#submitBtns").attr("disabled",true);
                            $("#againBtns").attr("disabled",true);
                        }
                        $("#theGetMarks").text(data.score);
                    }
                },
                error:function(e){
                    alert(e.responseText);
                }
            })
            for (var i = 0; i < chapterArr.length; i++) {
                var flag = false;
                for (var j = 0; j < chapterArr[i].length; j++) {
                    if (chapterArr[i][j].testId == id1) {
                        teststr = chapterArr[i][j].testDetail;
                        var jsonstr =teststr;
                        $("#testPaper").empty();
                        $.each(jsonstr, function (i, value) {//取每一道大题信息
                            arrys.push(value.smallDetail.length);//记录每道大题的小题数目
                            arryTypes.push(value.testBigType);//记录每道大题的题型
                            score.push(value.smallScore);//记录每道大题的小题分值
                            $.each(value.smallDetail, function (k, hhh) {//取每道小题,将每道小题答案放入trueAnswer中
                                trueAnswer.push(hhh.trueAnswer);
                            })
                            var str1 = '<div class="oneBigType" id="oneBigType' + (i + 1) + '">' +
                                "</div>";
                            $("#testPaper").append(str1);
                            $("#oneBigType" + (i + 1)).empty();
                            if (value.testBigType == "choose") {
                                $("#oneBigType" + (i + 1)).append('<textarea style="border-style:none;display:block;font-size:17px;color:black;background-color:rgb(251, 247, 247);" readonly rows="1" cols="130">' + (i + 1) + ".选择题 共" + value.smallNum + "小题，每道小题" + value.smallScore + "分,总计" + value.allScore + "分</textarea>");
                                $.each(value.smallDetail, function (j, val) {//value.smallDetail为每道大题的小题集合,val为每道小题
                                    var str2 = '<div class="smallChooseType">' +
                                        '<textarea style="border-style:none;display:block;" readonly rows="1" cols="130">' + "(" + (j + 1) + ")" + val.title + "</textarea>" +
                                        '<textarea style="border-style:none;display:block;" readonly rows="1" cols="130">A.' + val.A + "</textarea>" +
                                        '<textarea style="border-style:none;display:block;" readonly rows="1" cols="130">B.' + val.B + "</textarea>" +
                                        '<textarea style="border-style:none;display:block;" readonly rows="1" cols="130">C.' + val.C + "</textarea>" +
                                        '<textarea style="border-style:none;display:block;" readonly rows="1" cols="130">D.' + val.D + "</textarea>" +
                                        '<div><span style="font-size:15px;color:grey;">你的答案：</span><input type="text" class="answers" id="' + (i + 1) + "_" + (j + 1) + '"style="margin-left:10px;border-left-style:none;border-right-style:none;border-top-style:none;border-bottom-color:black;border-bottom-width:thin;background-color:rgb(251, 247, 247);"/>' + '<img class="correctImg" id="img' + (i + 1) + "_" + (j + 1) + '">' + "</div>" +
                                        "</div>";
                                    $("#oneBigType" + (i + 1)).append(str2);
                                })
                            }
                            else if (value.testBigType == "judge") {
                                $("#oneBigType" + (i + 1)).append('<textarea style="border-style:none;display:block;font-size:17px;color:black;background-color:rgb(251, 247, 247);" readonly rows="1" cols="130">' + (i + 1) + ".判断题 共" + value.smallNum + "小题，每道小题" + value.smallScore + "分,总计" + value.allScore + "分</textarea>");
                                $.each(value.smallDetail, function (j, val) {
                                    var str2 = '<div class="smallJudgeType">' +
                                        '<textarea style="border-style:none;display:block;" readonly rows="1" cols="130">' + "(" + (j + 1) + ")" + val.title + "</textarea>" +
                                        '<div style="margin-top:15px;"> ' +
                                        '<input type="radio" class ="answerRadio" name="' + (i + 1) + "_" + (j + 1) + '"  value="true"/>对&nbsp&nbsp&nbsp ' +
                                        '<input type="radio" class ="answerRadio" name="' + (i + 1) + "_" + (j + 1) + '" value="false" />错' + '<img class="correctImg" id="img' + (i + 1) + "_" + (j + 1) + '">' +
                                        "</div>" +
                                        "</div>";
                                    $("#oneBigType" + (i + 1)).append(str2);
                                })
                            }
                            else {
                                $("#oneBigType" + (i + 1)).append('<textarea style="border-style:none;display:block;font-size:17px;color:black;background-color:rgb(251, 247, 247);" readonly rows="1" cols="130">' + (i + 1) + ".填空题 共" + value.smallNum + "小题，每道小题" + value.smallScore + "分,总计" + value.allScore + "分</textarea>");
                                $.each(value.smallDetail, function (j, val) {
                                    var tip = "";
                                    if (val.answer == "number")
                                        tip = "数字";
                                    else
                                        tip = "专有名词";
                                    var str2 = '<div class="smallJudgeType">' +
                                        '<textarea style="border-style:none;display:block;" readonly rows="1" cols="130">' + "(" + (j + 1) + ")" + val.title + "</textarea>" +
                                        '<div><span style="font-size:17px;color:grey;">你的答案：</span><input type="text" class="answers" id="' + (i + 1) + "_" + (j + 1) + '"style="margin-left:10px;border-left-style:none;border-right-style:none;border-top-style:none;border-bottom-color:black;border-bottom-width:thin;background-color:rgb(251, 247, 247);"/></div>&nbsp&nbsp&nbsp' + '<img class="correctImg" id="img' + (i + 1) + "_" + (j + 1) + '"/>'+
                                        "提示:(" + tip + ")" +
                                        "</div>";
                                    $("#oneBigType" + (i + 1)).append(str2);
                                    $(".correctImg").hide();
                                })
                            }
                        });
                        $("#testTitles").text(chapterArr[i][j].testName);
                        $("#testTitle2").text(chapterArr[i][j].testName);
                        $("#valueTry").text(chapterArr[i][j].submitCount + "次，取最高分为最终有效成绩");
                        $("#limitTimes").text(chapterArr[i][j].LimitTime);
                        $("#testIntroduce").text(chapterArr[i][j].testIntro);
                        flag = true;
                        break;
                    }
                }
                if (flag == true)
                    break;
            }
            $(".testrough").hide();
            $(".testentry").show();
        })
        $(document).on("click", ".begintest", function () {
            $(".testentry").hide();
            $("#gobackrough").hide();
            $(".testread").show();
        })
        $(document).on("click", "#submitBtns", function () {
            var marks = 0;
            var count=0;
           // alert("arrys.length:"+arrys.length+"||arrys[0]:"+arrys[0]);
            for (var i = 0; i < arrys.length; i++)
            {//arrys.length为大题数，arrys.[i]为第i+1道大题的小题数,存储的是length
                for (var j = 0; j < arrys[i]; j++)
                {
                    var ids = (i + 1) + "_" + (j + 1);
                    var yourAnswer = "";
                    if (arryTypes[i] == "judge") {
                        yourAnswer = $('input[name="' + ids + '"]:checked').val();
                    }
                    else {
                        yourAnswer = $("#" + ids).val();
                    }
                    count++;
                   // alert("yourAnswer:" + yourAnswer + "||trueAnswer:" + trueAnswer[count - 1]);
                    if (yourAnswer == trueAnswer[count - 1]) {
                        marks += parseInt(score[i]);
                        $("#img" + ids).attr("src", "./img/right.jpg");
                        $("#img" + ids).css("width", "20px;");
                       // $("#img" + ids).show();
                    }
                    else {
                        $("#img" + ids).attr("src", "./img/wrong.jpg");
                        $("#img" + ids).css("width", "20px;");
                       // $("#img" + ids).show();
                    }
                }
            }
            $("#theGetMarks").text(marks);
            $("#againBtns").attr("disabled",false);
            $.ajax({
                type: "GET",
                url: "submitTest?t=" + new Date().getTime(),
                data:
                    {
                        "studentId": $("#studentIds").val(),
                        "score": marks,
                        "testId": $("#testIds").val(),
                    },
                cache: false,
                dataType: "json",
                success:function(data){
                    var sb="#testIds";
                    var id1="#submitCount"+$(sb).val();
                    var submitCount=$(id1).html();
                    var submitC=submitCount-data.submitCount;
                   // alert("data.submitCount:"+data.submitCount+"||"+"submitCount:"+submitCount);
                    if(submitC<=0)//达到最大提交次数，则显示对错情况
                    {
                        $(".correctImg").show();
                        $("#submitBtns").attr("disabled",true);
                        $("#againBtns").attr("disabled",true);
                    }
                    alert("提交成功,您还有"+submitC+"次提交机会");
                },
                error:function(e){
                    alert(e.responseText);
                }
            })
            $("#submitBtns").attr("disabled",true);
        })
        $(document).on("click", "#againBtns", function () {
            $("#submitBtns").attr("disabled",false);
            $("#againBtns").attr("disabled",true);
            $("#theGetMarks").empty();
            $(".correctImg").hide();
            $(".answers").val("");
            $(".answerRadio").attr("checked",false);
        })
    })
    $(document).on("click", "#ownIndex", function () {
        if ( studentId != "-1") {
            window.location.href = "students.jsp";
        }
        else {
            if ( teacherId != "-1") {
                window.location.href = "teacher.jsp";
            }
            else {
                alert("a unexpected error happened!");
            }
        }
    });
    $(document).on("click", ".topicDetail", function (e) {//点击话题详情，进入话题讨论页面,并用txt文件统计浏览次数
        var v_id=e.target.id;
        var topicId=v_id.substr(5,3);//获取topicId
        $("#topicArea").hide();
        var topicInfo="";
        $.each(topicArry,function(i,val){
            if(val.topicId==topicId)
            {
                //alert("val.topicId:"+val.topicId);
                topicInfo=val;
                //alert(topicInfo.topicTime);
            }
        })
        $("#topicDetail").html(topicInfo.topicDetail);
        $("#topicWriterName").html(topicInfo.topicWriterName);
       $("#topicWriterImg").attr("src","./img/"+topicInfo.topicWriterImg);
        $("#topicTime").html(topicInfo.topicTime);
        $("#topicId").attr("value",topicInfo.topicId);
        $("#topicWritterType").html(topicInfo.writterType);
        $.ajax({
            type:"GET",
            url:"showComment",
            cache:false,
            dataType:"json",
            data:{"topicId":$("#topicId").val()},
            success:function(data){
                var strhtm="";
                var strh
                $("#topicCommentDivs").empty();
                $.each(data,function(i,val){
                   var writterInfo= changeToHtmlStr(val.writterInfo);
                    //alert(val.writterImg);
                    if(val.ReplyToId=="-1") {//属于评论类型的则用如下样式， 回复类型 用另一样式
                        strhtm = "<div class=\"topicCommentDiv\" >\n" +
                            "                    <div id=\"commentBar\" style=\"height:70px;width:100%;\">\n" +
                            "                        <div class=\"commentBarLeft\" style=\"float:left;width:50%;height:inherit;\">\n" +
                            "                            <div class=\"leftImg\" style=\"float:left;width:9%;height:70%;\">\n" +
                            '                            <img src="img/' + val.writterImg + '" style="width:100%;height:100%;border-radius:50%;\">\n' +
                            "                            </div>\n" +
                            '                          <div class=\"rightName\" style=\"float:left;width:49%;height:auto;\"><span style="float:left;height:auto;width:40%;"><strong>' + val.writterName + "</strong>&nbsp<span style=\"background-color: #e4b9c0;color:white;\">"+val.writterTypeInfo+'</span></span> <span style="width:50%;height:auto;float:left;margin-left:5px;color:darkgray;">'+val.writterTime+"</span>日发表</div>\n" +
                            "                        </div>\n" +
                            '                      <div class=\"commentBarRight\" style=\"float:left;color:'+val.isAlreadyPraise+';width:35%;height:inherit;\"><span style=\"float:right;margin-left:40px;\" class="praise" id="praise'+val.commentId+'">赞:'+val.PraiseNum + '</span> <span id="reply'+val.commentId+'" class="reply">回复:'+val.ReplyNum+"</span>"+'<span style="margin-right:30px;float:right;color:grey;" class="report" id="report'+val.commentId+'">举报（'+val.reportNum+")</span></div>\n"+
                            "                    </div>\n" +
                            "                    <div class=\"topicCommentDetail\"  style=\"text-align: left;margin-left:2%;width:98%;height:auto;word-break: break-all; word-wrap:break-word;\">\n" +
                            '                      <p style=\"white-space: pre-wrap;word-wrap:break-word;\">' + writterInfo + '</span></p>\n' +
                            "                    </div>\n" +
                            "                </div>\n" +
                            "                <hr>";
                        //alert(val.isAlreadyPraise);
                        $("#topicCommentDivs").append(strhtm);
                    }
                    else//如果是回复类型
                    {
                        var replyToInfo=changeToHtmlStr(val.replyToInfo);
                        strhtm="<div class=\"topicCommentDiv\" >\n" +
                            "                    <div id=\"commentBar\" style=\"height:70px;width:100%;\">\n" +
                            "                        <div class=\"commentBarLeft\" style=\"float:left;width:50%;height:inherit;\">\n" +
                            "                            <div class=\"leftImg\" style=\"float:left;width:9%;height:70%;\">\n" +
                            '                            <img src="img/' + val.writterImg + '" style="width:100%;height:100%;border-radius:50%;\">\n' +
                            "                            </div>\n" +
                            '                          <div class=\"rightName\" style=\"float:left;width:49%;height:auto;\"><span style="float:left;height:auto;width:40%;"><strong>' + val.writterName + "</strong>&nbsp<span style=\"background-color: #e4b9c0;color:white\">"+val.writterTypeInfo+'</span></span> <span style="width:50%;height:auto;float:left;margin-left:5px;color:darkgray;">'+val.writterTime+"</span>日发表</div>\n" +
                            "                        </div>\n" +
                            '                      <div class=\"commentBarRight\" style=\"float:left;color:'+val.isAlreadyPraise+';width:35%;height:inherit;\"><span style=\"float:right;margin-left:40px;\" class="praise" id="praise'+val.commentId+'">赞:'+val.PraiseNum + '</span> <span id="reply'+val.commentId+'" class="reply">回复:'+val.ReplyNum+"</span>"+'<span style="margin-right:30px;float:right;color:grey;" class="report" id="report'+val.commentId+'">举报（'+val.reportNum+")</span></div>\n"+
                            "                    </div>\n" +
                            "                    <div class=\"topicCommentDetail\"  style=\"text-align: left;margin-left:2%;width:98%;height:auto;word-break: break-all; word-wrap:break-word;\">\n" +
                            '                      <p style=\"white-space: pre-wrap;word-wrap:break-word;\">' + val.writterInfo + "<span style=\"color:mediumpurple\"> //回复@</span><span style=\"color:black\"><strong>"+val.replyToName+"</strong></span>:"+"<span style=\"color:darkgrey;\">"+replyToInfo+"</span></p>\n" +
                            "                    </div>\n" +
                            "                </div>\n" +
                            "                <hr>";
                        $("#topicCommentDivs").append(strhtm);

                    }
                })
            },
            error:function(e) {
                alert(e.responseText);
            }
        })
        $(".topicCommentArea").show();
        //浏览次数+1
        $.ajax({
            type:"GET",
            url:"readCount",
            cache:false,
            dataType:"json",
            data:{"topicId":$("#topicId").val()},
            success:function(data){
               // alert(data.result);
            },
            error:function(e){
                alert(e.responseText);
            }
        })


    })
    $(document).on("click", ".reply", function (e) {
        var v_id=e.target.id;
        var commentId=v_id.substr(5,3);//获取commentId
       // alert("commentId:"+commentId);
        var time=new Date().Format("yyyy-MM-dd HH:mm:ss");
        alert("time:"+time);
        var commentInfo = prompt('回复他/她:');
        if(commentInfo!=""&&commentInfo!=null)
        {
            $.ajax({
                type:"GET",
                url:"writeComment",
                data:{"replyToId":commentId,
                    "commentInfo":commentInfo,
                    "belongTopicId":$("#topicId").val(),
                    "commentWritterType":$("#commentWritterType").val(),
                    "commentWritterId":$("#commentWritterId").val(),
                    "commentTime":time
                },
                dataType:"json",
                cache:false,
                success:function(data){
                    if(data.result=="forbiden"){
                        alert("您现在正处于禁言中，无法发表评论!");
                    }
                    else{
                        alert("回复成功!");
                    }
                },
                error:function(e){
                    alert(e.responseText);
                }
            })
        }
        else alert("回复内容不能为空!");
    });
    $(document).on("click", ".report", function (e) {
        var v_id=e.target.id;
        var commentId=v_id.substr(6,3);//获取commentId
        alert("report"+commentId);
            $.ajax({
                type:"GET",
                url:"reportComment",
                data:{
                    "commentId":commentId
                },
                dataType:"json",
                cache:false,
                success:function(data){
                    alert("举报成功!");
                },
                error:function(e){
                    alert(e.responseText);
                }
            })
    });
    $(document).on("click", ".praise", function (e) {
        var v_id=e.target.id;
        var commentId=v_id.substr(6,3);//获取commentId
            $.ajax({
                type: "GET",
                url: "praise?t=" + new Date().getTime(),
                data: {
                    "commentId": commentId,
                    "praiseUserType": $("#commentWritterType").val(),
                    "praiseUserId": $("#commentWritterId").val()
                },
                dataType: "json",
                cache: false,
                success: function (data) {
                    var id = "praise" + commentId;
                    var praiseNum = $("#" + id).text().substr(2, 5);
                    var tex = $("#" + id).text();
                    // alert("text:"+tex);
                    // alert("praiseNum:"+praiseNum);
                    if (data.isPraise == "ok") {
                        var count = parseInt(praiseNum) + 1;
                        $("#" + id).html("赞:" + count);
                        $("#" + id).css("color", "orange");
                        //alert("praise");
                    }
                    else if (data.isPraise == "alreadyPraise") {
                        //alert("alreadyPraise");
                        var count = parseInt(praiseNum) - 1;
                        $("#" + id).html("赞:" + count);
                        $("#" + id).css("color", "grey");
                    }
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            });
    });

    $(document).on("click", "#emotion", function () {
        $("#imgChooseDiv").show();
    });
    $(document).on("click", "#personalInfo", function () {
        alert("hello,you want to personalInfo");
    });
    $(document).on("click", "#exit", function () {
        $.ajax({
            type: "GET",
            url: "exitSystem?t=" + new Date().getTime(),
            data: { "exit": "true" },
            dataType: "json",
            cache: false,
            success: function (data) {
                window.location.reload();
                window.location.href = "index.jsp";
            },
            error: function (xhr) {
                alert(xhr.responseText);
            }
        });
    });
    $(function () {
        $("#beginStudy").click(function () {
            $("#lists div").eq($("#courseware").index()).removeClass("whites");
            $("#lists div").eq($("#courseware").index()).addClass("greens").siblings().removeClass("greens");
            $("#lists div").eq($("#courseware").index()).siblings().addClass("whites");
            $(".noticedisplay").hide();
            $(".evaluationdisplay").hide();
            $(".coursewaredisplay").show();
            $(".testdisplay").hide();
            $(".examdisplay").hide();
            $(".discussdisplay").hide();

        })
        $(".imgChoose").click(function(){
            var id=$(this).attr("id");
            var src="img/"+id.substr(6,18);
            var strhtm='<img style="width:25px;height:25px;" src="'+src+'">';
            $("#writeComment").append(strhtm);
            $("#imgChooseDiv").hide();
        })
        $("#writeCommentButton").click(function(){
            var commentInfo;
                commentInfo=$("#writeComment").html();
            commentInfo=changeToDBstr(commentInfo);
            var time=new Date().Format("yyyy-MM-dd HH:mm:ss");
            if(commentInfo!=""&&commentInfo!=null)
            {
                $.ajax({
                    type:"GET",
                    url:"writeComment",
                    data:{"replyToId":"-1",
                        "commentInfo":commentInfo,
                        "belongTopicId":$("#topicId").val(),
                        "commentWritterType":$("#commentWritterType").val(),
                        "commentWritterId":$("#commentWritterId").val(),
                        "commentTime":time
                    },
                    dataType:"json",
                    cache:false,
                    success:function(val){
                        if(val.result=="forbiden"){
                            alert("您现在正处于禁言中，无法发表评论!");
                        }
                        else{
                            alert("评论成功!");
                            $("#writeComment").html("");
                            lastLocation=0;
                            commentDetail="";
                            hasEmotion=false;
                            var writterInfo=changeToHtmlStr(val.writterInfo);
                            var strhtm="<div class=\"topicCommentDiv\" >\n" +
                                "                    <div id=\"commentBar\" style=\"height:70px;width:100%;\">\n" +
                                "                        <div class=\"commentBarLeft\" style=\"float:left;width:50%;height:inherit;\">\n" +
                                "                            <div class=\"leftImg\" style=\"float:left;width:9%;height:70%;\">\n" +
                                '                            <img src="img/' + val.writterImg + '" style="width:100%;height:100%;border-radius:50%;\">\n' +
                                "                            </div>\n" +
                                '                          <div class=\"rightName\" style=\"float:left;width:49%;height:auto;\"><span style="float:left;height:auto;width:40%;"><strong>' + val.writterName + "</strong>&nbsp<span style=\"background-color: #e4b9c0;color:white;\">"+val.writterTypeInfo+'</span></span> <span style="width:50%;height:auto;float:left;margin-left:5px;color:darkgray;">'+val.writterTime+"</span>日发表</div>\n" +
                                "                        </div>\n" +
                                '                      <div class=\"commentBarRight\" style=\"float:left;color:'+val.isAlreadyPraise+';width:35%;height:inherit;\"><span style=\"float:right;margin-left:40px;\" class="praise" id="praise'+val.commentId+'">赞:'+val.PraiseNum + '</span> <span id="reply'+val.commentId+'" class="reply">回复:'+val.ReplyNum+"</span>"+'<span style="margin-right:30px;float:right;color:grey;" class="report" id="report'+val.commentId+'">举报（'+val.reportNum+")</span></div>\n"+
                                "                    <div class=\"topicCommentDetail\"  style=\"text-align: left;margin-left:2%;width:98%;height:auto;word-break: break-all; word-wrap:break-word;\">\n" +
                                '                      <p style=\"white-space: pre-wrap;word-wrap:break-word;\">' + writterInfo + '</span></p>\n' +
                                "                    </div>\n" +
                                "                </div>\n" +
                                "                <hr>";
                            $("#topicCommentDivs").append(strhtm);
                        }
                    },
                    error:function(e){
                        alert(e.responseText);
                    }
                })
            }
            else alert("回复内容不能为空!");
        });
        $("#writeTopicButton").click(function(){
            var TopicInfo=$("#writeTopic").val();
            var time=new Date().Format("yyyy-MM-dd HH:mm:ss");
            if(TopicInfo!=""&&TopicInfo!=null)
            {
                $.ajax({
                    type:"GET",
                    url:"writeTopic",
                    data:{
                        "topicDetail":TopicInfo,
                        "belongCourseId":courseids,
                        "topicWritterType":$("#commentWritterType").val(),
                        "topicWritterId":$("#commentWritterId").val(),
                        "topicTime":time
                    },
                    dataType:"json",
                    cache:false,
                    success:function(data){
                        if(data.isWrite=="ok")
                        alert("writeTopic success");
                        else  alert("writeTopic fail");

                    },
                    error:function(e){
                        alert(e.responseText);
                    }
                })
            }
            else alert("回复内容不能为空!");
        });
        var reportLastDate;
        var isFistrClickRep=true;
        $("#reportButton").click(function(){//1分钟之内不能再次举报
            if(isFistrClickRep)//如果是第一次按
            {
                var date=new Date();
                reportLastDate=date;//记录按下的时间
                //发送给后台
                $.ajax({
                    type:"GET",
                    url:"reportTopic",
                    data:{
                        "topicId":$("#topicId").val()
                    },
                    dataType:"json",
                    cache:false,
                    success:function(data){
                        alert("举报成功!");
                    },
                    error:function(e){
                        alert(e.responseText);
                    }
                })
                isFistrClickRep=false;
            }
            else
            {
                var date=new Date();
                var nowHour=date.getMinutes();
                if(nowHour==reportLastDate.getMinutes())
                {
                    alert("您已经举报过了");
                }
                else
                {
                    reportLastDate=date;
                    //发送给后台
                    $.ajax({
                        type:"GET",
                        url:"reportTopic",
                        data:{
                            "topicTd":$("#topicId").val()
                        },
                        dataType:"json",
                        cache:false,
                        success:function(data){
                            alert("举报成功!");
                        },
                        error:function(e){
                            alert(e.responseText);
                        }
                    })
                }

            }


        });
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
</script>

<style>
    .vjs-default-skin .vjs-big-play-button {left:40%;top:40%;}
    div{
        white-space:nowrap;
    }
</style>
<body style="overflow-x:hidden">
<input type="hidden" id="commentWritterId" value="<%=commentWritterId%>">
<input type="hidden" id="commentWritterType" value="<%=commentWritterType%>">
<div id="imgStr" style="display:none;"></div>
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
            <li><a href="../数据统计/count.html">数据统计</a></li>
        </ul>
    </div>
    <div class="rightofhead">
        <div style="width:80%;float: left;height:inherit;">
            <input class="text" value="在这里搜索课程" onfocus="if (value =='在这里搜索课程'){value =''}" onblur="if (value ==''){value='在这里搜索课程'}" />
            <input type="button" class="searchbutton" value="搜索" id="button1">
        </div>
        <div style="width:20%;float:left;height:inherit;padding-top:4px;padding-right:0px;display:none;" id="afterLogin" data-toggle="popover" title="???" data-placement="bottom" >
            <img src="." style="height:30px;width:30px;border-radius:15px;border-style:none;" id="Pic"/>
        </div>
    </div>
</div>


<div class="title">
    <img  style="height:inherit;float:left;width:170px;margin-left:10px;" id="schoolSrc">
    <div class="coursename">
        <span style="margin-left:25px;color:black;font-size:18px;display:block;"><a id="hreftointro" href=""><strong id="coursename"></strong></a></span>
        <div style="color:limegreen;margin-top:10px;" id="teacher"><span>吕布</span><span>典韦</span><span>赵云</span><span>关羽</span><span>张飞</span><span>马超</span></div>
    </div>
    <span style="color:forestgreen;float:right;margin-top:44px;text-align:center;font-size:15px;">评价课程</span>
</div>


<div class="functionlist">
    <div style="width:90%;margin-left:5%;margin-right:5%;height:170px;"><img style="border-radius: 15px;margin-top:10px;margin-left:10px;width:90%;" id="coursePostSrc"></div>
    <div style="width:100%;height:auto;" id="lists">
        <div class="whites" id="notice">公告</div>
        <div class="whites">评分标准</div>
        <div class="whites" id="courseware">课件</div>
        <div class="whites">作业</div>
        <div class="whites">考试</div>
        <div class="whites">讨论区</div>
    </div>
</div>
<div class="displays">
    <div class="noticedisplay">
        <div class="entry">
            <div style="width:60%;float:left;height:auto;">
                <h3><strong style="color:black;">亲爱的<%=studentName%></strong></h3>
                <h4 id="welcomeName"></h4>
            </div>
            <input type="button" value="开始学习" style="width:20%;margin-left:110px;margin-top:30px;height:35px;text-align:center;padding-top:6px;color:white;background:rgba(175,130,18,0.88);float:left;border-radius: 4px;border-style:none;" id="beginStudy">
        </div>
        <div class="noti"><span style="margin-left:3px;">公告</span></div>
        <div style="width:80%;margin-left:10%;margin-right:10%;height:1px;border-bottom-style:ridge;border-bottom-color:rgba(232,225,225,1.00);border-bottom-width: thin;"></div>
        <div class="nonenotice"><strong>暂时没有公告</strong></div>
        <div class="noticedetail">

        </div>

    </div>
    <div class="evaluationdisplay">
        <h3 style="margin-left:22px;">评分标准</h3>
        <div style="height:1px;width:94%;margin-left:3%;margin-right:3%;border-bottom-style:solid;border-bottom-color:rgba(231,224,224,1.00);border-bottom-width: thin;"></div>
        <textarea style="font-size:16px;color:black;border-style:none;" readonly name="evaluationdetail" cols="130" rows="7" id="evaluationdetail"></textarea>
    </div>
    <div class="coursewaredisplay">
        <div class="coursewaretitle">课件<input type="button" value="回到目录总览" id="gobackutil" style="margin-left:30px;background:white;color:black;text-align:center;width:13%;border-style:none;display:none;">
        </div>
        <div class="coursewaredetail"><!--总的章节总览-->
        </div>
        <!--视频播放部分-->
        <div class="videoview">
            <div class="studyWare" style="float:left;width:60%;height:auto;margin-left:3%;margin-top:30px;">
                <div class="textDetail" style="width:90%;height:auto;margin-left:3%;" >
                    <div >
                        <a  href="./texts/text1.txt" download="text1.txt">下载课件text1.txt</a>
                    </div>
                </div>
                <div class="videos" >
                </div>
            </div>
            <div class="categories">
                <div class="group">
                    <div class="categorytitle">第1章 the power of the language </div>
                    <div class="categorydetails">
                        <div class="categorydetail">1.1 welcome to League of legends</div>
                        <div class="categorydetail">1.1 welcome to League of legends</div>
                    </div>
                </div>

            </div>
        </div>
        <!--视频播放部分结束-->
    </div>
    <div class="testdisplay">
        <div class="testrough">
        </div>
        <div class="testentry">
            <div class="testentrytitle"><span id="testTitles"></span>
                <img src="./img/t01ef39187ba1e08a27.png" style="height:25px;margin-left:410px;"><span id="gobackrough" style="color:orange;"><input type="button" value="返回" style="border-style:none;background:inherit;width:60px;height:auto;padding:2px;"></span>
            </div>
            <div class="introducetest">
                <span style="display:block;">1.可尝试次数：<span style="color:red;"><strong id="valueTry"></strong></span></span>
                <span style="display:block;margin-top:17px;">2.测验限时：<span style="color:red;"><strong id="limitTimes">20分钟</strong></span></span>
            </div>
            <span style="display:block;width:95%;" id="testIntroduce"></span>
            <input type="checkbox" checked="checked" id="undertaking"><span style="color:dodgerblue;font-size:17px;"><strong>依照学术诚信条款，我保证本测验答案是我独立完成的。</strong></span>
            <input type="button" value="开始做题" id="begintests"  class="begintest">
        </div>
        <div class="testread">
            <h3 id="testTitle2" style="display:inline;"></h3>
            <img src="./img/t01ef39187ba1e08a27.png" style="height:25px;margin-left:410px;"><span id="gobackentry" style="color:orange;"><input type="button" value="返回" style="border-style:none;background:inherit;width:60px;height:auto;padding:2px;"></span>
            <hr />
            <form  style="width:100%;height:auto;">
                <input type="hidden" name="studentId" id="studentIds" value="<%=studentId%>" />
                <input type="hidden" name="courseId" id="courseIds" />
                <input type="hidden" name="testId" id="testIds" />
                <div id="testPaper" style="width:100%;height:auto;margin-bottom:20px;background-color:rgb(251, 247, 247);">
                </div>
                <input type="button" value="重做" id="againBtns" style="border-style:none;border-radius:4px;background-color:greenyellow;width:103px;height:33px;padding-top:6px;text-align:center;margin-left:12px;" disabled/>
                <input type="button" value="提交" id="submitBtns" style="border-style:none;border-radius:4px;background-color:greenyellow;width:103px;height:33px;padding-top:6px;text-align:center;margin-left:12px;"/>
                <span style="margin-left:150px;" >总成绩得分：</span><span style="font-size:23px;color:red;" id="theGetMarks"></span>
            </form>
        </div>
    </div>




























    <div class="examdisplay">
        <div id="examEntry" style="width:80%;margin-left:10%;height:auto;">
            <div id="examTitle"><strong></strong></div>
            <div>考试说明:</div>
            <p id="examSuplement" style="word-break: break-all">
                本次考试为本门课程最终期末考试，请同学们在指定时间内限时完成，拒绝作弊，诚信考试
            </p>
            <div >限制时间：<span id="limitTime" style="color:green">60</span>分钟</div>
            <div >开始时间：<span id="startTime" style="color:green">2019-5-23</span></div>
            <div >截止时间：<span id="endTime" style="color:green">2019-6-23</span></div>
            <div class="ExamState" style="background-color: papayawhip;color:red"> 状态：已完成，已截止</div>
            <input type="button" value="开始考试" id="beginExam">
        </div>
        <div id="examDetail" style="display:none;">
            <div id="CountMsg" class="HotDate" style="color:red;">
                <h3 style="color:red;">倒计时：</h3>
                <span id="t_d">00天</span>
                <span id="t_h">00时</span>
                <span id="t_m">00分</span>
                <span id="t_s">00秒</span>
            </div>
            <form method="post" enctype="multipart/form-data" target="hidden_frame" action="submitExam">
                <input type="hidden" name="courseId" id="examPramOfCourseId">
            <div id="oneExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第一大题：
                    <p id="oneProblemTitle" style="word-break: break-all">

                    </p>
                </div>
                <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                    <div  style="width:inherit;height:95%;" id="problemAnswerSrcLookThroughOne"></div>
                    <div><input type="file" class="chooseFile" name="oneProblemSrc" id="chooseFileOne"></div>
                </div>
            </div>
            <div id="twoExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第二大题：
                    <p id="twoProblemTitle" style="word-break: break-all">

                    </p>
                </div>
                <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                    <div  style="width:inherit;height:95%;" id="problemAnswerSrcLookThroughTwo"></div>
                    <div style="height:auto;"><input type="file" class="chooseFile" name="twoProblemSrc" id="chooseFileTwo"></div>
                </div>
            </div>
            <div id="threeExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第三大题：
                    <p id="threeProblemTitle" style="word-break: break-all">

                    </p>
                </div>
                <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                    <div  style="width:inherit;height:95%;" id="problemAnswerSrcLookThroughThree"></div>
                    <div style="height:auto;"><input type="file" class="chooseFile" name="threeProblemSrc" id="chooseFileThree"></div>
                </div>
            </div>
            <div id="fourExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第四大题：
                    <p id="fourProblemTitle" style="word-break: break-all">

                    </p>
                </div>
                <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                    <div  style="width:inherit;height:95%;" id="problemAnswerSrcLookThroughFour"></div>
                    <div style="height:auto;"><input type="file" class="chooseFile" name="fourProblemSrc" id="chooseFileFour"></div>
                </div>
            </div>
            <div id="fiveExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第五大题：
                    <p id="fiveProblemTitle" style="word-break: break-all">

                    </p>
                </div>
                <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                    <div  style="width:inherit;height:95%;" id="problemAnswerSrcLookThroughFive"></div>
                    <div style="height:auto;"><input type="file" class="chooseFile" name="fiveProblemSrc" id="chooseFileFive"></div>
                </div>
            </div>
                <iframe name="hidden_frame" style="display:none"></iframe>
                <div><input type="submit" value="提交测试" id="submitExamBtn" onclick="return submitForm();"></div>
            </form>
        </div>

    </div>
    <div class="discussdisplay">
        <div class="coursewaretitle">
            讨论区
        </div>
        <div style="height:1px;width:94%;margin-left:3%;margin-right:3%;border-bottom-style:solid;border-bottom-color:rgba(231,224,224,1.00);border-bottom-width: thin;margin-bottom:30px;"></div>
        <span style="margin-left:2%;width:95%;">欢迎大家来到讨论区！本讨论区供各位同学就课程问题进行交流学习。请同学们认真阅读下面的【讨论区使用规则】，然后再进行相关发表，谢谢！</span>
        <div style="height:1px;width:94%;margin-left:3%;margin-right:3%;border-bottom-style:solid;border-bottom-color:rgba(231,224,224,1.00);border-bottom-width: thin;margin-bottom:30px;"></div>
        <div id="topicArea">
            <textarea id="writeTopic" rows="6" cols="100"></textarea>
        <input type="button" value="发起话题" id="writeTopicButton"  class="begindiscuss">
        <div style="height:1px;width:94%;margin-left:3%;margin-right:3%;border-bottom-style:solid;border-bottom-color:rgba(231,224,224,1.00);border-bottom-width: thin;margin-bottom:10px;margin-top:10px;"></div>
        <div class="sorts">
            按<span style="margin-left:10px;color:darkgrey;font-size:15px;">话题最新</span><span style="margin-left:10px;color:darkgrey;font-size:15px;">浏览最多</span><span style="margin-left:10px;color:darkgrey;font-size:15px;">评论最多</span>
        </div>
        <div class="discussrough">
            <div class="adiscussrough">
                <span>hello</span>
                <span><span id="discussername1" style="color:green;font-size:15px;display:inline;">sbsbsb</span><span style="color:darkgrey;display:inline;font-size:15px;margin-left:15px;">于2016年4月5日发表</span>
				<span style="color:green;display:inline;font-size:14px;margin-left:15px;" id="discussdetail1">详情</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:470px;">浏览:34</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">评论:9</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">举报(0)</span></span>
            </div>
            <div class="adiscussrough">
                <span>hello</span>
                <span><span id="discussername2" style="color:green;font-size:15px;display:inline;">sbsbsb</span><span style="color:darkgrey;display:inline;font-size:15px;margin-left:15px;">于2016年4月5日发表</span>
				<span style="color:green;display:inline;font-size:14px;margin-left:15px;" id="discussdetail2">详情</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:470px;">浏览:34</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">评论:9</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">举报(0)</span></span>
            </div>
            <div class="adiscussrough">
                <span>hello</span>
                <span><span id="discussername3" style="color:green;font-size:15px;display:inline;">sbsbsb</span><span style="color:darkgrey;display:inline;font-size:15px;margin-left:15px;">于2016年4月5日发表</span><span style="color:green;display:inline;font-size:14px;margin-left:15px;" id="discussdetail3">详情</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:470px;">浏览:34</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">评论:9</span>
				<span style="color:darkgrey;display:inline;font-size:14px;margin-left:15px;">举报(0)</span></span>
            </div>






        </div>
        </div>
        <div class="topicCommentArea" style="display:none;width:inherit;height:auto;">
            <input type="hidden" value="" id="topicId">
            <div style="width:100%;height:auto;background-color:ivory;">
            <div id="titleTop" style="height:12%;width:100%;">
                <div id="titleImg" style="float:left;width:8%;">
                <img id="topicWriterImg" src="img/2F2B3E11578DC027FBC1B022EAA126C8.jpg" style="height:100%;width:100%;border-radius:50%; ">
                </div>
                <div id="titleRight" style="float:left;width:50%;">
                    <div id="titleRightName"> <span style="color:red;margin-left:10px;" id="topicWriterName">马云</span>&nbsp<span id="topicWritterType" style="background-color: #e4b9c0;color:white;"></span></div>
                    <div style="margin-top:30px;">于<span id="topicTime" style="color:greenyellow">2019/2/1</span>发表：</div>
                </div>
                <div style="float:left;width:39%;height:inherit;">
                    <div style="width:10%;margin-right:5%;float:right;"><span style="cursor: pointer;color:red;" id="reportButton">举报(恶意话题)</span></div>
                </div>
            </div>
            <div id="topicTitle" style="font-size:25px;color:black;height:auto;margin-top:3%;margin-bottom:10%;width:100%;text-align: center;">
                <h3 id="topicDetail">二十天上王者</h3>
            </div>
                <div style="height:40px;width:100%;"></div>
            </div>
            <hr>
            <div id="writeComment" contenteditable="true" style="width:60%;height:90px;border:solid;border-color:black;border-width: 1px;word-break:break-all;"></div>
            <div style="margin-top:5px;">
                <div style="width:100%;height:auto;display:inline;">
                    <span style="height:auto;width:auto;float:left;"><input type="button" id="writeCommentButton" value="发表评论" style="background-color:darkolivegreen;text-align: center;border-radius:2px;border:none;"></span>
                    <span style="height:25px;width:25px;float:left;" id="emotion"><img style="height:inherit;width:inherit;" src="img/emotionLogo.jpg"></span>
                    <div id="imgChooseDiv" style="margin-left:10%;width:30%;height:auto;display:none;border-style:solid;border-width:1px;border-color:goldenrod">
                        <div><span id="emtionbishi.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/bishi.jpg" ></span>
                        <span id="emtionfanu.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/fanu.jpg" ></span>
                        <span id="emtionkonghuang.png" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/konghuang.png" ></span>
                        <span id="emtionxiaoku.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/xiaoku.jpg" ></span>
                            </div>
                        <div><span id="emtionkaixin.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/kaixin.jpg" ></span>
                            <span id="emtionbeishang.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/beishang.jpg" ></span>
                            <span id="emtionhuaji.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/huaji.jpg" ></span>
                            <span id="emtiondaku.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/daku.jpg" ></span>
                        </div>
                        <div>
                        <span id="emtionweixiao.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/weixiao.jpg" ></span>
                        <span id="emtionaixin.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/aixin.jpg" ></span>
                            <span id="emtionbaiyan.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/baiyan.jpg" ></span>
                            <span id="emtiontiaopi.jpg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/tiaopi.jpg" ></span>
                        </div>
                        <div>
                            <span id="emtionkuxiao.jpeg" class="imgChoose"><img style="width:7%;margin-left:2%;height:4%;" src="img/kuxiao.jpeg" ></span>
                        </div>
                    </div>

            </div>
            <hr>
            <!--评论区样式-->
            <div id="topicCommentDivs" class="topicCommentDivs" style="width:100%;height:auto;margin-top:50px;">
                <div class="topicCommentDiv" >
                <div id="commentBar" style="height:70px;width:100%;">
                    <div id="commentBarLeft" style="float:left;width:50%;height:inherit;">
                        <div id="leftImg" style="float:left;width:9%;height:70%;">
                            <img src="img/2F2B3E11578DC027FBC1B022EAA126C8.jpg" style="width:100%;height:100%;border-radius:50%;">
                        </div>
                        <div id="rightName" style="float:left;width:49%;height:auto;"><h5><strong>放逐之刃</strong></h5></div>
                    </div>
                    <div id="commentBarRight" style="float:left;width:35%;height:inherit;"><span style="margin-right:5px;float:right;color:grey;">赞:554</span></div>
                </div>
                <div class="topicCommentDetail" id="topicCommentDetail" style="text-align: left;margin-left:2%;width:100%;height:auto;word-break: break-all; word-wrap:break-word;">
                    <p style="white-space: pre-wrap;word-wrap:break-word;"></p>
                </div>
            </div>
                <hr>
                <div class="topicCommentDiv" >
                    <div class="commentBar" style="height:70px;width:100%;">
                        <div class="commentBarLeft" style="float:left;width:50%;height:inherit;">
                            <div class="leftImg" style="float:left;width:9%;height:70%;">
                                <img src="img/2F2B3E11578DC027FBC1B022EAA126C8.jpg" style="width:100%;height:100%;border-radius:50%;">
                            </div>
                            <div class="rightName" style="float:left;width:49%;height:auto;"><h5><strong>放逐之刃</strong></h5></div>
                        </div>
                        <div class="commentBarRight" style="float:left;width:35%;height:inherit;"><span style="margin-right:5px;float:right;color:grey;">赞:554</span></div>
                    </div>
                    <div class="topicCommentDetail"  style="text-align: left;margin-left:2%;width:100%;height:auto;word-break: break-all; word-wrap:break-word;">
                        <p style="white-space: pre-wrap;word-wrap:break-word;">aiwehgiuhawefiauwhegiuhweifuhwuiegweh
                            <span style="color:mediumpurple"> //@</span><span style="color:black"><strong>马云</strong></span>:
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
</div>
<script type="text/javascript">
    var interval=null;
    var hasSubmit=false;
    var firstClick=true;
    var EndTime;
    var limitTime;
    $(document).ready(function(){
        $(".chooseFile").change(function(){
            var fileId=$(this).attr("id");
            var lookThroughImgId="lookThroughImg"+fileId.substr(10,6);
            var problemAnswerSrcLookThroughId="problemAnswerSrcLookThrough"+fileId.substr(10,6);
            //alert(problemAnswerSrcLookThroughId);
           // alert(lookThroughImgId);
            var strhtm='<img style="width:65%;height:inherit;margin-left:20%;" id="'+lookThroughImgId+'"/>';
            $("#"+problemAnswerSrcLookThroughId).html(strhtm);
            var pic = document.getElementById(lookThroughImgId),
                file = document.getElementById(fileId);

            var ext=file.value.substring(file.value.lastIndexOf(".")+1).toLowerCase();

            // gif在IE浏览器暂时无法显示
            if(ext!='png'&&ext!='jpg'&&ext!='jpeg'){
                alert("图片的格式必须为png或者jpg或者jpeg格式！");
                return;
            }
            var isIE = navigator.userAgent.match(/MSIE/)!= null,
                isIE6 = navigator.userAgent.match(/MSIE 6.0/)!= null;

            if(isIE) {
                file.select();
                var reallocalpath = document.selection.createRange().text;

                // IE6浏览器设置img的src为本地路径可以直接显示图片
                if (isIE6) {
                    pic.src = reallocalpath;
                }else {
                    // 非IE6版本的IE由于安全问题直接设置img的src无法显示本地图片，但是可以通过滤镜来实现
                    pic.style.filter = "progid:DXImageTransform.Microsoft.AlphaImageLoader(sizingMethod='image',src=\"" + reallocalpath + "\")";
                    // 设置img的src为base64编码的透明图片 取消显示浏览器默认图片
                    pic.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==';
                }
            }else {
                html5Reader(file,lookThroughImgId);
            }
        })
    })
    function html5Reader(file,lookThroughImgId) {
        var file = file.files[0];
        var reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function (e) {
            var pic = document.getElementById(lookThroughImgId);
            pic.src = this.result;
        }
    }


    function getRTime(){
        var NowTime = new Date();
        var t = EndTime.getTime() - NowTime.getTime();
        /*var d=Math.floor(t/1000/60/60/24);
        t-=d*(1000*60*60*24);
        var h=Math.floor(t/1000/60/60);
        t-=h*60*60*1000;
        var m=Math.floor(t/1000/60);
        t-=m*60*1000;
        var s=Math.floor(t/1000);*/

        var d=Math.floor(t/1000/60/60/24);
        var h=Math.floor(t/1000/60/60%24);
        var m=Math.floor(t/1000/60%60);
        var s=Math.floor(t/1000%60);

        document.getElementById("t_d").innerHTML = d + "天";
        document.getElementById("t_h").innerHTML = h + "时";
        document.getElementById("t_m").innerHTML = m + "分";
        document.getElementById("t_s").innerHTML = s + "秒";
        if(d==0&&h==0&&m==0&&s==0){
            clearInterval(interval);
            if(hasSubmit==false) {
                alert("考试结束！您已无法再进行答题，请点击'提交测试'按钮进行提交");
                $(".chooseFile").attr("disabled", true);
            }
        }
    }
</script>
</body>
</html>

