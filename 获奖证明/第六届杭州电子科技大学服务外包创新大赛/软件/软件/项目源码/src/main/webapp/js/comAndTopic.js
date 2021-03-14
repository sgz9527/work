$(document).ready(function(){
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
                    alert("评论成功");
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
                        "</div>"+
                        "                    <div class=\"topicCommentDetail\"  style=\"text-align: left;margin-left:2%;width:98%;height:auto;word-break: break-all; word-wrap:break-word;\">\n" +
                        '                      <p style=\"white-space: pre-wrap;word-wrap:break-word;\">' + writterInfo + '</span></p>\n' +
                        "                    </div>\n" +
                        "                </div>\n" +
                        "                <hr>";
                    $("#topicCommentDivs").append(strhtm);

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
            alert("topicId:"+$("#topicId").val());
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
})
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
$(document).on("click", "#emotion", function () {
        $("#imgChooseDiv").show();
});
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
                alert("reply success");
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