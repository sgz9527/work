$(document).ready(function(){
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
                    //alert("chapter.length:"+chapterArr.length);
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
                            strh+='</span></div><input type="button" value="查看" class="gotofinish" id="'+v.testId+'"></div>';
                            strh+='<div class="testdetail">'+
                                '<span>发布时间:'+v.testStartTime+"</span>"+
                                "<span>截止时间:"+v.testEndTime+"</span>"+
                                '<span>总分合计:'+v.testAllMark+"分</span>"+
                                '<div><span style="float:left;">有效提交次数：</span>'+'<span id="submitCount'+v.testId+'" style="float:left;" >'+v.submitCount+"</span>"+'<span style="float:left;">次,取最高分数为有效成绩</span></div>'+
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
                url: "getCourse?t=" + new Date().getTime(),
                data: { "courseid": courseids },
                dataType: "json",
                cache: false,
                success: function (data) {
                    //如果试卷已经发布
                    if(data.examPass==1){
                        //如果是当前登录教师所发布的试卷，则显示其他教师的审核信息
                        if (data.examPublishTeacherId == teacherId) {
                            $("#checkInfoDetail").empty();
                            $.each(data.checkInfo,function(i,value){
                                var strhtm='<div class="oneCheckInfo">\n' +
                                    '                审核教师：<span  style="color:yellowgreen">'+value.checkTeacher+'</span>&nbsp&nbsp 审核建议：<span  style="color:yellowgreen">'+value.checkSuplement+'</span> 结果:<span  style="color:orangered">'+value.checkResult+'</span>\n' +
                                    '            </div>';
                                $("#checkInfoDetail").append(strhtm);
                            })
                            $("#checkInfo").show();
                            $("#lookExam").hide();
                            $("#deleteExam").hide();
                        }
                        //显示试卷信息
                        $("#examSuplement").html(data.examSuplement);
                        $("#examStartTime").val(data.examStartTime);
                        $("#examEndTime").val(data.examEndTime);
                        $("#examLimitTime").val(data.examLimitTime);
                        $("#oneProblemTitle").html(data.examOneTitle);
                        $("#twoProblemTitle").html(data.examTwoTitle);
                        $("#threeProblemTitle").html(data.examThreeTitle);
                        $("#fourProblemTitle").html(data.examFourTitle);
                        $("#fiveProblemTitle").html(data.examFiveTitle);
                        $("#problemAnswerSrcLookThroughOne").html(data.examOneAnswer);
                        $("#problemAnswerSrcLookThroughTwo").html(data.examTwoAnswer);
                        $("#problemAnswerSrcLookThroughThree").html(data.examThreeAnswer);
                        $("#problemAnswerSrcLookThroughFour").html(data.examFourAnswer);
                        $("#problemAnswerSrcLookThroughFive").html(data.examFiveAnswer);
                        $("#examDesignArea").show();
                        $("#submitExamTeacherBtn").attr("disabled", true);
                        $("#submitExamTeacherBtn").hide();
                        //显示已提交学生列表
                        $("#examBody").empty();
                        $.each(data.studentExamList,function(i,value){
                            var str1=' <tr >\n' +
                                '                   <td class="studentExamName" id="studentExamName'+value.studentId+'">'+value.studentName+'</td>\n' +
                                '                   <td class="studentExamAccount" id="studentExamAccount'+value.studentId+'">'+value.studentAccount+'</td>\n' +
                                '                   <td class="studentExamType" id="studentExamType'+value.studentId+'">期末测试</td>\n' +
                                '               </tr>';
                            $("#examBody").append(str1);
                        })

                    }
                    //说明试卷并未发布成功，有修改的机会
                    else {
                        //说明有教师进行了试卷设计
                        if (data.examPublishTeacherId != -1) {
                            //如果当前登录教师不是发布试卷的教师，则显示试卷信息
                            if (data.examPublishTeacherId != teacherId) {
                                //显示试卷信息
                                $("#examSuplement").html(data.examSuplement);
                                $("#examStartTime").val(data.examStartTime);
                                $("#examEndTime").val(data.examEndTime);
                                $("#examLimitTime").val(data.examLimitTime);
                                $("#oneProblemTitle").html(data.examOneTitle);
                                $("#twoProblemTitle").html(data.examTwoTitle);
                                $("#threeProblemTitle").html(data.examThreeTitle);
                                $("#fourProblemTitle").html(data.examFourTitle);
                                $("#fiveProblemTitle").html(data.examFiveTitle);
                                $("#problemAnswerSrcLookThroughOne").html(data.examOneAnswer);
                                $("#problemAnswerSrcLookThroughTwo").html(data.examTwoAnswer);
                                $("#problemAnswerSrcLookThroughThree").html(data.examThreeAnswer);
                                $("#problemAnswerSrcLookThroughFour").html(data.examFourAnswer);
                                $("#problemAnswerSrcLookThroughFive").html(data.examFiveAnswer);
                                $("#examDesignArea").show();
                                $("#submitExamTeacherBtn").attr("disabled", true);
                                $("#submitExamTeacherBtn").hide();
                                $("#checkDivs").show();
                            }
                            //否则显示审核列表
                            else {
                                //显示试卷信息
                                $("#examSuplement").html(data.examSuplement);
                                $("#examStartTime").val(data.examStartTime);
                                $("#examEndTime").val(data.examEndTime);
                                $("#examLimitTime").val(data.examLimitTime);
                                $("#oneProblemTitle").html(data.examOneTitle);
                                $("#twoProblemTitle").html(data.examTwoTitle);
                                $("#threeProblemTitle").html(data.examThreeTitle);
                                $("#fourProblemTitle").html(data.examFourTitle);
                                $("#fiveProblemTitle").html(data.examFiveTitle);
                                $("#problemAnswerSrcLookThroughOne").html(data.examOneAnswer);
                                $("#problemAnswerSrcLookThroughTwo").html(data.examTwoAnswer);
                                $("#problemAnswerSrcLookThroughThree").html(data.examThreeAnswer);
                                $("#problemAnswerSrcLookThroughFour").html(data.examFourAnswer);
                                $("#problemAnswerSrcLookThroughFive").html(data.examFiveAnswer);
                                $("#checkInfoDetail").empty();
                                $.each(data.checkInfo,function(i,value){
                                    var strhtm='<div class="oneCheckInfo">\n' +
                                        '                审核教师：<span  style="color:yellowgreen">'+value.checkTeacher+'</span>&nbsp&nbsp 审核建议：<span  style="color:yellowgreen">'+value.checkSuplement+'</span> 结果:<span  style="color:orangered">'+value.checkResult+'</span>\n' +
                                        '            </div>';
                                    $("#checkInfoDetail").append(strhtm);
                                })
                                $("#checkInfo").show();
                            }
                        }
                        //否则说明并没有教师进行试卷设计
                        else {
                            $("#setExam").show();
                        }
                    }
                },
                error: function (xhr) {
                    alert(xhr.responseText);
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
                            '\t\t\t\t<input type="button" style="color:green;display:inline;font-size:14px;margin-left:15px;border-style: none;" id="topic'+val.topicId+'"class="topicDetail" value="详情"/></div>\n' +
                            '\t\t\t\t<div style=\"float:left;width:64%;\"><span style=\"color:darkgrey;display:inline;font-size:14px;margin-left:470px;\">浏览:'+val.readCount+"</span>\n" +
                            "\t\t\t\t<span style=\"color:darkgrey;display:inline;font-size:14px;margin-left:15px;\">评论:"+val.comNum+"</span>\n" +
                            "\t\t\t\t<span style=\"color:darkgrey;display:inline;font-size:14px;margin-left:15px;\">举报("+val.reportNum+")</span></div></div>\n" +
                            "            </div>";
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
    $(document).on("click", "#ownIndex", function () {
        if (studentId != "-1") {
            window.location.href = "students.jsp";
        }
        else {
            if (teacherId != "-1") {
                window.location.href = "teacher.jsp";
            }
            else {
                alert("a unexpected error happened!");
            }
        }
    });
})