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
                                    alert("明明是courseIntroduce.jsp");
                                    str2 = '<div style="width:30%;margin-left:2%;height:210px;float:left;" class="oneSearchCourse" id="oneSearchCourse' + data[3 * i + j].courseId + '">' +
                                        '<a href="ourseIntroducce.jsp?courseid=' + data[3 * i + j].courseId + '">' + '<div style="width:100%;height:160px;background-image:url(' + "'" + "./img/" + data[3 * i + j].courseImgSrc + "'" + ');background-repeat:no-repeat;background-size:cover;" class="courseBac" id="courseBac' + data[3 * i + j].courseId + '">' +
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
                                    alert("明明是courseIntroduce.jsp");
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
})