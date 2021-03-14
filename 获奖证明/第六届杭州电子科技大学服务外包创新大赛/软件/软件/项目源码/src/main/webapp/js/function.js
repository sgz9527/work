
$(document).ready(function () {
    $.ajax({
        type: "GET",
        data:{"type":"all"},
        url: "/getAllCourse?t=" + new Date().getTime(),
        dataType: "json",
        cache: false,
        success: function (data) {
            // alert("success:  " + data.length + "门课程");
            var strhtml="";
            $(".allcourse").empty();
            $.each(data, function (j, vals) {
                // alert(vals.courseId);
                strhtml+='<a href="courseIntroduce.jsp?courseid='+vals.courseId+'">'+
                    '<div class="coursedisplay" id="coursedisplay'+(j+1)+'">'+
                    '<div class="displayleft" style="position:relative;"><img src="./img/'+vals.courseImgSrc+'"style="height:inherit;">'+
                    '<span style="position:absolute;left:2px;top:2px;background-color:rgba(0,0,0,0.5);color:white;font-size:15px;">'+vals.courseState+"</span>"+
                    "</div>"+
                    '<div class="displayright">'+
                    '<span style="color:black;font-size:20px;display:block;"><strong>'+vals.courseName+"</strong></span>"+
                    '<div class="schandteac"><span>'+vals.school+'</span><span style="margin-left:30px;">'+vals.teachers +"</span></div>"+
                    '<div class="introducecourse">'+vals.courseIntroduce+"</div>"+
                    '<div style="width:100%;height:auto;margin-top:20px;">'+
                    '<span style="color:orange;">'+vals.fans+"</span>人正在学习"+" &nbsp&nbsp&nbsp&nbsp&nbsp进行到第"+'<span style="color:orange;">'+vals.Progress+"</span>周"+
                    "</div>"+
                    "</div>"+
                    "</div>"+
                    "</a>";
            })
            $(".allcourse").html(strhtml);
        },
        error:function(xhr){
            alert(xhr.responseText);
        }
    })
    $.ajax({
        type: "GET",
        data:{"type":"all"},
        url: "/getTopTenCourse?t=" + new Date().getTime(),
        dataType: "json",
        cache: false,
        success: function (data) {
            // alert("success:  " + data.length + "门课程");
            var strhtml=" <img src=\"./img/49E89BEA3F1B3F1AC788F5F94C4A457F.png\">\n" +
                "<span style=\"color:white;font-size:20px;display: block;\"><strong>本周最热课程</strong></span>";
            $(".fixright").empty();
            $.each(data, function (j, vals) {
                if(j<=9) {
                    if (j == 0) {
                        strhtml += "<div class=\"numberone\">\n" +
                            "        <div class=\"leftone\" style=\"background-image:url(../img/"+vals.courseImgSrc+')">\n' +
                            "            <span style=\"color:white;font-size:22px;margin-left:1px;margin-top:1px;\"><strong>1</strong></span>\n" +
                            "        </div>\n" +
                            "        <div class=\"rightone\">\n" +
                            "            <div style=\"color:red;font-size:16px;\" ><a style=\"color:red;\" href=\"courseIntroduce.jsp?courseid="+vals.courseId+'"><strong>'+vals.courseName+"</strong></a></div>\n" +
                            "            <div style=\"color:white;margin-top:45px;\"><span style=\"color:orange;font-size:16px;\"><strong>"+vals.fans+"</strong></span>人在学</div>\n" +
                            "        </div>\n" +
                            "    </div>";
                           // var urls="../img/"+vals.courseImgSrc;
                           // $(".leftone").css("background-image","url("+urls+")");
                    }
                    else {
                        strhtml +="<div style=\"width:100%;height:auto;text-align: left;margin-left:2px;\">\n" +
                            "        <span style=\"color:white;font-size:16px;\" href=\"courseIntroduce.jsp?courseid="+vals.courseId+'"><strong>'+(j+1)+".</strong></span>\n" +
                            "        <span><a style=\"color:white;font-size:18px;\" href=\"courseIntroduce.jsp?courseid="+vals.courseId+'">'+vals.courseName+"</a></span>\n" +
                            "    </div>";
                    }
                }
            })
            $(".fixright").html(strhtml);
        },
        error:function(xhr){
            alert(xhr.responseText);
        }
    })
    $("#all").addClass("clicgreen");
    $(".type tr td").click(function () {
        $(".type tr").children().removeClass("clicgreen");
        $(this).addClass("clicgreen").siblings().removeClass("clicgreen");
        $("#types").text("->" + $(this).text());
        var datas = $(this).text();
        if (datas == "全部")
            datas = "all";
        $.ajax({
            type: "GET",
            data: { "type": datas },
            url: "/getAllCourse?t=" + new Date().getTime(),
            dataType: "json",
            cache: false,
            success: function (data) {
                // alert("success:  " + data.length + "门课程");
                var strhtml = "";
                $(".allcourse").empty();
                $.each(data, function (j, vals) {
                    // alert(vals.courseId);
                    strhtml += '<a href="courseIntroduce.jsp?courseid=' + vals.courseId + '">' +
                        '<div class="coursedisplay" id="coursedisplay' + (j + 1) + '">' +
                        '<div class="displayleft" style="position:relative;"><img src="./img/' + vals.courseImgSrc + '"style="height:inherit;">' +
                        '<span style="position:absolute;left:2px;top:2px;background-color:rgba(0,0,0,0.5);color:white;font-size:15px;">' + vals.courseState + "</span>" +
                        "</div>" +
                        '<div class="displayright">' +
                        '<span style="color:black;font-size:20px;display:block;"><strong>' + vals.courseName + "</strong></span>" +
                        '<div class="schandteac"><span>' + vals.school + '</span><span style="margin-left:30px;">' + vals.teachers + "</span></div>" +
                        '<div class="introducecourse">' + vals.courseIntroduce + "</div>" +
                        '<div style="width:100%;height:auto;margin-top:20px;">' +
                        '<span style="color:orange;">'+vals.fans+'</span>人正在学习 &nbsp&nbsp&nbsp&nbsp&nbsp进行到第<span style="color:orange;">'+vals.Progress+"</span>周" +
                        "</div>" +
                        "</div>" +
                        "</div>" +
                        "</a>";
                })
                $(".allcourse").html(strhtml);
            },
            error: function (xhr) {
                alert(xhr.responseText);
            }
        })


    });
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
    alert("hello,you want to personalInfo");
});
$(document).on("click", "#exit", function () {
    $.ajax({
        type: "GET",
        url: "exitSystem?t=" + new Date().getTime(),
        dataType: "json",
        cache: false,
        success: function (data) {
            window.location.href = "index.jsp";
        },
        error: function (xhr) {
            alert(xhr.responseText);
        }
    });
});