var utilId=-1;
$(document).ready(function() {
    courseids = getUrlParam("courseid");
    $("#courseIds").val(courseids);
    $("#afterLogin").show();
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
        data: {"courseid": courseids},
        dataType: "json",
        success: function (data) {
            $("title").text(data.courseName + "-管理页面");
            $("#coursePostSrc").attr("src", "./img/" + data.courseImgSrc);
            $("#coursename").text(data.courseName);
            $("#welcomeName").text("欢迎来到《" + data.courseName + "》");
            $("#schoolSrc").attr("src", "./img/" + data.comPic);
        },
        error: function (xhr) {
            alert(xhr.responseText);
        }
    })
    $("#lists div").eq($("#notice").index()).removeClass("whites");
    $("#lists div").eq($("#notice").index()).addClass("greens").siblings().removeClass("greens");
    $("#lists div").eq($("#notice").index()).siblings().addClass("whites");
    $(".noticedisplay").show();
    $(".evaluationdisplay").hide();
    $(".coursewaredisplay").hide();
    $(".testdisplay").hide();
    $(".examdisplay").hide();
    $(".discussdisplay").hide();
    $.ajax({
        type: "GET",
        url: "chapterinfo?t=" + new Date().getTime(),
        data: {"courseid": courseids},
        cache: false,
        dataType: "json",
        success: function (data) {
            $(".coursewaredetail").empty();
            $("#chapter").empty();
            $(".categories").empty();
            $("#testbelongutil").empty();
            $.each(data, function (i, values) {
                var sbs = [];
                var cou = i + 1;
                var strhtml1 = "";
                var str = "";
                var strhtml3 = "";
                if (i == 0)
                    str = '<option value="' + values.chapterId + '" selected>' + values.titleNumberName + "&nbsp" + values.titleName + '</option>';
                else
                    str = '<option value="' + values.chapterId + '">' + values.titleNumberName + "&nbsp" + values.titleName + '</option>';
                strhtml1 = '<div class="courseutil" id="' + values["chapterId"] + '">' +
                    '<div class="utiltitle" id="utiltitle' + values["chapterId"] + '">' +
                    '<img src="./img/t01501c515ccc7294fe.png" style="height:15px;margin-left:5px;" id="logo' + values["chapterId"] + '">&nbsp' + values["titleNumberName"] + '<span style="margin-left:14px;">' + values["titleName"] + "</span>" +
                    "</div>" +
                    '<div class="utildetails" id="utildetails' + values["chapterId"] + '">' +
                    "</div>" + "</div>";
                strhtml3 = '<div class="group" id="group' + values.chapterId + '">\n' +
                    '<div class="categorytitle">' + values.titleNumberName + "&nbsp" + values.titleName + "</div>\n" +
                    '<div class="categorydetails" id="categorydetails' + values.chapterId + '">\n' +
                    "</div>\n" +
                    "</div>";
                $("#testbelongutil").append(str);
                $(".categories").append(strhtml3);
                $(".coursewaredetail").append(strhtml1);
                $.ajax({
                    type: "GET",
                    url: "utilinfo?t=" + new Date().getTime(),
                    data: {"chapterId": values["chapterId"]},
                    cache: false,
                    dataType: "json",
                    success: function (data) {
                        var str = "";
                        var strhtml2 = "";
                        $.each(data, function (j, vals) {
                            strhtml2 = '<div class="utildetail" id="' + vals.utilId + '">' + vals.utilDetailTitle + "&nbsp" + vals.utilDetailName +
                                '<span style="margin-left:50px;color:greenyellow;font-size:15px;">共' + vals.utilDetailVideoNumber + "个视频</span>" +
                                '<span style="margin-left:20px;color:greenyellow;font-size:15px;">' + vals.utilDetailTextNumber + "份文档</span>" +
                                "</div>";
                            str = '<div class="categorydetail" id="util' + vals.utilId + '">' + vals.utilDetailTitle + vals.utilDetailName + "</div>\n";
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
        utilId = idr;
        $("#belongUtilId").val(utilId);
        $("#belongUtilId1").val(utilId);
        $(".categorydetail").css("background-color", "white");
        $("#util" + idr).css("background-color", "yellow");
        var strh = "";
        $.ajax({
            type: "GET",
            url: "getSource?t=" + new Date().getTime(),
            data: {"utilId": idr},
            cache: false,
            dataType: "json",
            success: function (data) {
                $(".videos").empty();
                $(".textDetail").empty();
                $.each(data, function (i, values) {
                    if (values.sourceType == "video") {
                        strh = '<div class="avideo">' +
                            '<video controls preload="auto" width="528px" height="300px"  poster="' + "./img/" + values.videoposter + '"' +
                            ' data-setup="{}">' + '<source src="' + "./video/" + values.sourceSrc + '" type="video/mp4" />' + "</video></div>";
                        //alert(strh);
                        $(".videos").append(strh);
                    }
                    else {
                        strh = " <div >\n" +
                            '<a  href="./Texts/' + values.sourceSrc + '" download="' + values.sourceSrc + '">下载课件' + values.sourceSrc + "</a>\n" +
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
        //$(".studyWare").show();
        $(".coursewaredetail").hide();
        $(".videoview").show();
        $("#gobackutil").show();
    })
    $(document).on("click", ".categorydetail", function () {
        $(".videos").empty();
        var idr = $(this).attr("id").substr(4, 6);
        utilId = idr;//把目前的utilId赋给公共变量utilId
        $("#belongUtilId").val(utilId);
        $("#belongUtilId1").val(utilId);
        $(".categorydetail").css("background-color", "white");
        $(this).css("background-color", "yellow");
        //alert(idr);
        var strh = "";
        $.ajax({
            type: "GET",
            url: "getSource?t=" + new Date().getTime(),
            data: {"utilId": idr},
            cache: false,
            dataType: "json",
            success: function (data) {
                $(".videos").empty();
                $(".textDetail").empty();
                $.each(data, function (i, values) {
                    if (values.sourceType == "video") {
                        strh = '<div class="avideo">' +
                            '<video controls preload="auto" width="528px" height="300px"  poster="' + "./img/" + values.videoposter + '"' +
                            ' data-setup="{}">' + '<source src="' + "./video/" + values.sourceSrc + '" type="video/mp4" />' + "</video></div>";
                        //alert(strh);
                        $(".videos").append(strh);
                    }
                    else {
                        strh = " <div >\n" +
                            '<a  href="./Texts/' + values.sourceSrc + '" download="' + values.sourceSrc + '">下载课件' + values.sourceSrc + "</a>\n" +
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
    $("#submitTextButton").click(function () {
        $("#submitTextDiv").show();
        $("#submitVideoDiv").hide();
    })
    $("#submitVideoButton").click(function () {
        $("#submitTextDiv").hide();
        $("#submitVideoDiv").show();
    })
    $("#submitTextButtonToServ").click(function () {
        $("#submitTextDiv").hide();
    })
    $("#submitVideoButtonToServ").click(function () {
        /* var formData = new FormData();
         formData.append('posterSrc', $("#posterSrc")[0].files[0]);
         formData.append('belongUtilId', $('#belongUtilId').val());
         formData.append('videoSrc', $("#videoSrc")[0].files[0]);
         formData.append('sourceType', $('#sourceType').val());
         $.ajax({
             url: "uploadVideo",
             type: 'POST',
             cache: false,
             data: formData,
             processData: false,
             contentType: false,
             success: function (data) {
                 alert(data.result);
             },
             error: function (e) {
                 alert(e.responseText);
             }

         })
     })*/
        $("#submitVideoDiv").hide();
    })
})
