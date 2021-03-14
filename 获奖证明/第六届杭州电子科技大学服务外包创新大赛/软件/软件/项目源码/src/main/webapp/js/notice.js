$(document).ready(function(){
    var notSave=0;
    var waitnot="";
    var finishnoticehtml="";
    var waitnoticehtml="";
    $.ajax({
        type: "GET",
        url: "noticeinfo?time=" + new Date().getTime(),
        cache: false,
        data:{"courseid":courseids},
        dataType: "json",
        success: function (data) {
            var waitprocounts=0;
            var finishprocounts=0;
            $("#alreadyprodiv").empty();
            $("#waitprodiv").empty();
            $("#alreadypro").css({
                background:"green",
                color:"white"
            })
            $("#waitpro").css({
                background:"white",
                color:"black"
            })
            $.each(data,function(i,values){
                if(values["noticeState"]=="finish"){
                    finishnoticehtml+='<div class="onenotice">'+
                        "<h3>"+values["noticeTitle"]+"</h3>"+
                        '<textarea style="color:darkgrey;border-style:none;background: rgba(248,222,222,0.1);" cols="120" rows="14" readonly id="textarea'+values.noticeId+'">'+
                        hh(values["noticeBody"])+"</textarea>"+
                        '<p style="text-align:right;width:100%;heigth:auto;">'+values["noticeWritter"] +values["noticeTime"]+"</p>"+
                        '<span style="color:rgba(231,227,227,0.5);background:rgba(0,0,0,0.4);width:30px;margin-left:80%;height:auto;border-radius: 2px;">'+values["noticeSystemTime"]+"</span></div>";
                }
                else{
                    waitnoticehtml+= '<div class="onenotice" id="waitnotice'+values.noticeId+'">'+
                        '<div><textarea style="color:black;border-style:none;background: rgba(248,222,222,0.1);font-size:25px;" cols="60" rows="1"  class="waittextareatitle" id="waittextareatitle'+values.noticeId+'" name="noticetitle" readonly>'+values["noticeTitle"]+"</textarea></div>"+
                        '<div><textarea style="color:darkgrey;border-style:none;background: rgba(248,222,222,0.1);" cols="120" rows="14"  class="waittextareabody" id="waittextareabody'+values.noticeId+'" name="noticebody" readonly >'+hh(values["noticeBody"])+
                        "</textarea><div>"+
                        '<div><span style="margin-left:45%"> 发布人：</span><input type="text" readonly value="'+values.noticeWritter+'" class="waitwritter" id="waitwritter'+values.noticeId+
                        '" name="writer"  style="margin-left:1px;border-style:none;outline: none;color:darkgrey;" ><span style="margin-left:1px"> 时间：</span> <input type="date" readonly value="'+values.noticeTime+'"class="waittime" id="waittime'+values.noticeId+
                        '" name="time"  style="border-style:none;outline: none;color:darkgrey;text-align: left;" ></div><div><input type="button" style="width:100px;height:30px;background:green;color:white;text-align:center;font-size:16px;margin-left:410px;border-style:none;border-radius: 5px;" value="删除" class="deletenotice" id="deletenotice'+values.noticeId+
                        '"><input type="button" style="width:100px;height:30px;background:green;color:white;text-align:center;font-size:16px;margin-left:20px;border-style:none;border-radius: 5px;" value="点击修改" class="updatenotice" id="updatenotice'+values.noticeId+
                        '"><input type="button" style="width:100px;height:30px;background:grey;color:white;text-align:center;font-size:16px;margin-left:20px;border-style:none;border-radius: 2px;" value="保存"  class="savenotice" id="savenotice'+values.noticeId+'" disabled><input type="button" style="width:100px;height:30px;background:green;color:white;text-align:center;font-size:16px;margin-left:20px;border-style:none;border-radius: 2px;" value="发布" class="proclaimnotice" id="proclaimnotice'+values.noticeId+
                        '"></div></div>';
                }
            })
            $("#alreadypro").css({
                background:"green"
            });
            // alert(waitnoticehtml);
            $("#alreadyprodiv").html(finishnoticehtml);
            $("#waitprodiv").html(waitnoticehtml);
            finishnoticehtml="";
            waitnoticehtml="";
            waitnot="true";
            $("#alreadyprodiv").show();
            $("#waitprodiv").hide();
            $("#noticeview").show();
            $(".nonenotice").hide();
        },
        error:function(xhr){
            alert(xhr.responseText);
            waitnot="none";
            $(".nonenotice").show();
            $("#noticeview").hide();
        }

    });
    $(document).on("click",".updatenotice",function(e){
        notSave++;
        $(this).attr("disabled",true);
        var ids=$(this).attr("id");
        var noticeId=ids.substring(12,16);
        $("#waittextareatitle"+noticeId).removeAttr("readonly");
        $("#waittextareatitle"+noticeId).css({
            outlineStyle:"solid",
            outlineColor:"orange",
            outlineWidth:"1px"
        })
        $("#waittextareabody"+noticeId).removeAttr("readonly");
        $("#waittextareabody"+noticeId).css({
            outlineStyle:"solid",
            outlineColor:"orange",
            outlineWidth:"1px"
        })
        $("#waitwritter"+noticeId).removeAttr("readonly");
        $("#waitwritter"+noticeId).css({
            outlineStyle:"solid",
            outlineColor:"orange",
            outlineWidth:"1px"
        })
        $("#waittime"+noticeId).removeAttr("readonly");
        // $("#waittime"+noticeId).attr("type","date");
        $("#waittime"+noticeId).css({
            outlineStyle:"solid",
            outlineColor:"orange",
            outlineWidth:"1px"
        })
        $("#savenotice"+noticeId).css(
            {
                background:"green"

            });
        $("#savenotice"+noticeId).removeAttr("disabled");
        $("#proclaimnotice"+noticeId).attr("disabled",true);
        $("#proclaimnotice"+noticeId).css(
            {
                background:"grey"

            });
    });
    $(document).on("click",".deletenotice",function(e){
        var ids=$(this).attr("id");
        var noticeId=ids.substring(12,16);
        $.ajax({
            type: "GET",
            url: "deleteNotice",
            cache: false,
            data:{"noticeId":noticeId
            },
            dataType: "json",
            success: function (data) {
                window.location.reload();
            },
            error:function(xhr){
                alert(xhr.responseText);
            }
        });
    });
    $(document).on("click",".savenotice",function(e){
        var ids=$(this).attr("id");
        var noticeId=ids.substring(10,14);
        notSave--;
        //alert($("#waittime"+noticeId).val());
        $.ajax({
            type: "GET",
            url: "updateNotice",
            cache: false,
            data:{"noticeId":noticeId,
                "time":$("#waittime"+noticeId).val(),
                "noticeTitle":$("#waittextareatitle"+noticeId).val(),
                "noticeDetail":$("#waittextareabody"+noticeId).val(),
                "noticeState":"wait",
                "writer": $("#waitwritter"+noticeId).val(),
                "belongCourseId":courseids
            },
            dataType: "json",
            success: function (data) {
                if(data.result=="ok"){
                    $("#waittextareatitle"+noticeId).attr("readonly","readonly");
                    $("#waittextareatitle"+noticeId).css({
                        outline:"none"
                    });
                    $("#waittextareabody"+noticeId).attr("readonly","readonly");
                    $("#updatenotice"+noticeId).attr("disabled",false);
                    $("#waittextareabody"+noticeId).css({
                        outline:"none"
                    });
                    $("#waitwritter"+noticeId).attr("readonly",true);
                    $("#waitwritter"+noticeId).css({
                        outline:"none"
                    });
                    $("#waittime"+noticeId).attr("readonly",true);
                    $("#waittime"+noticeId).css({
                        outline:"none"
                    });
                    $("#savenotice"+noticeId).css(
                        {
                            background:"grey"

                        });
                    $("#savenotice"+noticeId).attr("disabled",true);
                    $("#proclaimnotice"+noticeId).removeAttr("disabled");
                    $("#proclaimnotice"+noticeId).css(
                        {
                            background:"green"

                        });
                }
            },
            error:function(xhr){
                alert(xhr.responseText);
            }
        });

    })
    $(document).on("click",".proclaimnotice",function(e){
        var ids=$(this).attr("id");
        var noticeId=ids.substr(14,5);
        // alert($("#waittime"+noticeId).val());
        $.ajax({
            type: "GET",
            url: "updateNotice",
            cache: false,
            data:{"noticeId":noticeId,
                "time":$("#waittime"+noticeId).val(),
                "noticeTitle":$("#waittextareatitle"+noticeId).val(),
                "noticeDetail":$("#waittextareabody"+noticeId).val(),
                "noticeState":"finish",
                "writer": $("#waitwritter"+noticeId).val(),
                "belongCourseId":courseids
            },
            dataType: "json",
            success: function (data) {
                $("#"+ids).attr("disabled",true);
                $("#"+ids).css({
                    background:"grey"
                })
                $("#updatenotice"+noticeId).attr("disabled",true);
                $("#updatenotice"+noticeId).css({
                    background:"grey"
                })
                $("#deletenotice"+noticeId).attr("disabled",true);
                $("#deletenotice"+noticeId).css({
                    background:"grey"
                })
                alert("发布成功");
            },
            error:function(xhr){
                alert(xhr.responseText);
            }
        });
    })
    //公告按钮事件
    $("#alreadypro").click(function(){
        $("#alreadypro").css({
            background:"green",
            color:"white"
        });
        $("#waitpro").css({
            background:"white",
            color:"black"
        });
        $("#alreadyprodiv").show();
        $("#waitprodiv").hide();
    })
    $("#waitpro").click(function(){
        $("#waitpro").css({
            background:"red",
            color:"white"
        });
        $("#alreadypro").css({
            background:"white",
            color:"black"
        });
        $("#alreadyprodiv").hide();
        $("#waitprodiv").show();
    })
    $("#addnotice").click(function(){
        if(notSave==0||waitnot=="none")//如果没有未保存的或者没有待发布的公告则可以新建公告
        {
            $("#addnoticeview").show();
            $("#noticeview").hide();
            $(this).attr("disabled",true);
            $("#saveaddnotice").attr("disable",false);
            $(this).css({
                background:"grey"
            });
        }
        else{
            alert("有未保存的公告！");
        }
    })
    $("#gobacknoticeview").click(function(){
        $("#addnoticeview").hide();
        $("#noticeview").show();
        $("#addnotice").removeAttr("disabled");
        $("#saveaddnotice").attr("disable",true);
        $("#addnotice").css({
            background:"rgba(175,130,18,0.88)"
        });
    })
    $("#saveaddnotice").click(function(){
        $("#belongCourseId").val(courseids);
        $.ajax({
            type: "GET",
            url: "addSaveNotice",
            cache: false,
            data:$("#addNoticeForm").serialize(),
            dataType: "json",
            success: function (data) {
                if(data.result!=-1){
                    $("#proNoticeId").val(data.result);
                    $("#saveaddnotice").attr("disabled",true);
                    $("#saveaddnotice").css({
                        background:"grey"
                    })
                    $("#updateaddnotice").removeAttr("disabled");
                    $("#updateaddnotice").css({
                        background:"green"
                    })
                    $("#proaddnotice").css({
                        background:"green"
                    })
                    $("#proaddnotice").attr("disabled",false);
                }
            },
            error:function(xhr){
                alert(xhr.responseText);
            }
        });
        //alert("saveaddnotice");
        /*var ssss="sdZCFWdvsweagf<br/>大风车  <br/>已关闭及 符不通过    8互斥GV好吧";
        var getFormatCode= ssss.replace(/\r\n/g, '<br/>').replace(/\n/g, '<br/>').replace(/\s/g, ' ');                  由textarea转为可存储到数据库的字符
        var sbd=hh(ssss);由可存储到数据库的字符转为textarea
        $("#noticebody").text(sbd);
        alert(sbd);*/
    })
    $("#proaddnotice").click(function(){
        $("#belongCourseId").val(courseids);
        $.ajax({
            type: "GET",
            url: "proNotice",
            cache: false,
            data:{"noticeId":$("#proNoticeId").val()},
            dataType: "json",
            success: function (data) {
                if(data.result=="ok"){
                    $("#proaddnotice").attr("disabled",true);
                    $("#proaddnotice").css({
                        background:"grey"
                    })
                    $("#updateaddnotice").removeAttr("disabled");
                    $("#updateaddnotice").css({
                        background:"green"
                    })
                    $("#proaddnotice").css({
                        background:"green"
                    })
                    $("#proaddnotice").attr("disabled",false);
                }
            },
            error:function(xhr){
                alert(xhr.responseText);
            }
        });
    })
    $("#updateaddnotice").click(function(){
        $("#proaddnotice").css({
            background:"grey"
        })
        $("#proaddnotice").attr("disabled",true);
    })
})