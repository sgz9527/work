var bigTestTypeNum=0;
$(document).ready(function(){
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
    $("#gobackrough").click(function () {
        $(".testentry").hide();
        $(".testrough").show();
    })
    $("#gobackentry").click(function () {
        $(".testread").hide();
        $(".testentry").show();
        $("#gobackrough").show();
    })
    $("#prohomework").click(function () {
        $(".testread").hide();
        $(".testentry").hide();
        $(".testrough").hide();
        $(".testEdit").hide();
        $(".testWaitRough").hide();
        $(".prohomeworkDiv").show();
    })
    $("#homework").click(function () {
        $(".testread").hide();
        $(".testentry").hide();
        $(".testrough").show();
        $(".prohomeworkDiv").hide();
        $(".testEdit").hide();
        $(".testWaitRough").hide();
    })
    $("#waitProHomeWork").click(function () {
        chapterArr1 = [];
        $.ajax({
            type: "GET",
            url: "getWaitTest?t=" + new Date().getTime(),
            data: { "courseId": courseids },
            dataType: "json",
            cache: false,
            success: function (data) {
                //var strhtm='<div class="tests">'+
                //	'<div class="testtitle">'+data.chapterName+"</div>"+
                //"</div>";
                $.each(data, function (i, val) {
                    var flag = false;
                    for (var m = 0; m < chapterArr1.length; m++) {//将获取的json数组按每一章分开存储
                        if (chapterArr1[m][0].chapterId == val.chapterId) {
                            chapterArr1[m].push(val);
                            flag = true;
                            break;
                        }
                    }
                    if (flag == false) {
                        var arry = [];
                        arry.push(val);
                        chapterArr1.push(arry);
                    }

                })
                $(".testWaitRough").empty();
                $.each(chapterArr1, function (k, values) {//values是每一章对应的作业，有多个作业
                    var strhtm='<div class="waitTests" id="waitTests'+values[0].chapterId+'">'+
                        '<div class="waitTesttitle">'+values[0].chapterName+"</div>"+
                        "</div>";
                    $(".testWaitRough").append(strhtm);
                    $("#waitTests"+values[0].chapterId).empty();
                    $.each(values, function (n, v) {//每一章里的每一个作业
                        var strh = '<div class="waitTests">'+
                            '<div class="testtitle">'+v.chapterName+"</div>"+
                            '<div class="testdetails">'+
                            '<div class="testdetailtitle">'+'<div style="width:30%;height:auto;float:left;margin-top:7px;">'+v.testName+'<span style="color:white;background:rgba(151,146,146,0.5);font-size:15px;display:inline;">截止时间:'+v.testEndTime;
                            strh+='</span></div><input type="button" value="编辑作业" style="background-color: darkgreen;margin-right:5px;" class="EditTest" id="waitEditTest'+v.testId+'" ></div>';
                        strh+='<div class="testdetail">'+
                            '<span>发布时间:'+v.testStartTime+"</span>"+
                            "<span>截止时间:"+v.testEndTime+"</span>"+
                            '<span>总分合计:'+v.testAllMark+"分</span>"+
                            '<div><span style="float:left;">有效提交次数：</span>'+'<span id="submitCount'+v.testId+'" style="float:left;" >'+v.submitCount+"</span>"+'<span style="float:left;">次,取最高分数为有效成绩</span></div>'+
                            "</div>";
                        $("#waitTests" + values[0].chapterId).append(strh);
                        //alert(v.testDetail);
                    })

                })
            },
            error: function (xhr) {
                alert(xhr.responseText);
            }
        })
        $(".testread").hide();
        $(".testentry").hide();
        $(".testrough").hide();
        $(".testEdit").hide();
        $(".testWaitRough").show();
        $(".prohomeworkDiv").hide();
    })
    $("#addBigTestType").click(function(){
        if(bigTestTypeNum!=0){
            var ccc=confirm("点击添加后将不能对之前所创建大题的小题数目进行修改，确定添加？");
            if(!ccc){
                return false;
            }
        }
        var testId=$("#testIds").val();
        bigTestTypeNum++;
        var strhtml="";
        if($("#ProbremType").val()=="choose"){
            strhtml= '<div class="choose" id="'+bigTestTypeNum+"_choose"+'">'+
                '<div class="testnumber" style="font-size:17px;color:black;">第 <span id="testBigNumber'+bigTestTypeNum+'">'+
                "<strong>"+bigTestTypeNum+"</strong></span>题（选择题，共"+
                '<span id="'+bigTestTypeNum+"_smallNum1"+'">'+"0</span>"+"小题，每小题"+
                '<span id="'+bigTestTypeNum+"_smallScore1"+'">'+"1</span>"+"分）"+"总计："+
                '<span id="'+bigTestTypeNum+"_allScore1"+'">'+"0</span>"+"分"+"</div>"+
                '<input type="hidden" id="'+bigTestTypeNum+"_allScore"+'">'+
                '<input type="hidden" name="belongTestId" value="'+testId+'">'+
                '<input type="hidden" name="testProblemType" value="0">'+
                '<input type="hidden" name="testProblemTitle" id="'+bigTestTypeNum+"_title"+'">'+
                '<input type="hidden"  id="'+bigTestTypeNum+"_smallNum"+'">'+	'<input type="hidden" name="testProblemOrder" id="'+"testBigNumber"+bigTestTypeNum+'"'+'value="'+bigTestTypeNum+'">'+
                '<input type="hidden"  id="'+bigTestTypeNum+"_smallScore"+'">'+
                "</div>"+
                '<div class="addOneTestType">'+
                '<span style="font-size:17px;color:black;margin-left:2%;margin-top:10px;"'+">添加小题数目:&nbsp</span>"+
                '<input type="number" style="width:60px;margin-top:10px;" value="1" id="'+bigTestTypeNum+"_addSmallNum"+'" >'+'每小题分值:<input type="number" style="width:60px;margin-top:10px;" value="1" id="'+bigTestTypeNum+"_eachSmallScore"+'"'+">"+
                '<input type="button" value="添加" style="background:green;color:white;text-align:center;padding-top:5px;width:80px;height:32px;border-style:none;border-radius:3px;margin-left:3%;margin-top:5px;" id="'+"addSmallButtonChoose"+bigTestTypeNum+'" class="addSmallButtonChoose">'+
                "</div>";
            $(".writeTestDetail").append(strhtml);
        }
        else if($("#ProbremType").val()=="judge"){
            strhtml= '<div class="judge" id="'+bigTestTypeNum+"_judge"+'">'+
                '<div class="testnumber" style="font-size:17px;color:black;">第 <span id="testBigNumber'+bigTestTypeNum+'">'+
                "<strong>"+bigTestTypeNum+"</strong></span>题（判断题，共"+
                '<span id="'+bigTestTypeNum+"_smallNum1"+'">'+"0</span>"+"小题，每小题"+
                '<span id="'+bigTestTypeNum+"_smallScore1"+'">'+"1</span>"+"分）"+"总计："+
                '<span id="'+bigTestTypeNum+"_allScore1"+'">'+"0</span>"+"分"+"</div>"+
                '<input type="hidden" id="'+bigTestTypeNum+"_allScore"+'">'+
                '<input type="hidden" name="belongTestId" value="'+testId+'">'+
                '<input type="hidden" name="testProblemType" value="1">'+
                '<input type="hidden" name="testProblemTitle" id="'+bigTestTypeNum+"_title"+'">'+
                '<input type="hidden"  id="'+bigTestTypeNum+"_smallNum"+'">'+	'<input type="hidden" name="testProblemOrder" id="'+"testBigNumber"+bigTestTypeNum+'"'+'value="'+bigTestTypeNum+'">'+
                '<input type="hidden" id="'+bigTestTypeNum+"_smallScore"+'">'+
                "</div>"+
                '<div class="addOneTestType">'+
                '<span style="font-size:17px;color:black;margin-left:2%;margin-top:10px;">添加小题数目:&nbsp</span>'+
                '<input type="number" style="width:60px;margin-top:10px;" value="1" id="'+bigTestTypeNum+"_addSmallNum"+'" >'+
                '每小题分值:<input type="number" style="width:60px;margin-top:10px;" value="1" id="'+bigTestTypeNum+"_eachSmallScore"+'"'+'name="'+bigTestTypeNum+"_eachSmallScore"+'"'+">"+
                '<input type="button" value="添加" style="background:green;color:white;text-align:center;padding-top:5px;width:80px;height:32px;border-style:none;border-radius:3px;margin-left:3%;margin-top:5px;" id="'+"addSmallButtonJudge"+bigTestTypeNum+'" class="addSmallButtonJudge">'+
                "</div>";
            $(".writeTestDetail").append(strhtml);
        }
        else if($("#ProbremType ").val()=="fillin"){
            strhtml= '<div class="fillin" id="'+bigTestTypeNum+"_fillin"+'">'+
                '<div class="testnumber" style="font-size:17px;color:black;">第 <span id="testBigNumber'+bigTestTypeNum+'">'+
                "<strong>"+bigTestTypeNum+"</strong></span>题（填空题，共"+
                '<span id="'+bigTestTypeNum+"_smallNum1"+'">'+"0</span>"+"小题，每小题"+
                '<span id="'+bigTestTypeNum+"_smallScore1"+'">'+"1</span>"+"分）"+"总计："+
                '<span id="'+bigTestTypeNum+"_allScore1"+'">'+"0</span>"+"分"+"</div>"+
                '<input type="hidden" id="'+bigTestTypeNum+"_allScore"+'">'+
                '<input type="hidden" name="belongTestId" value="'+testId+'">'+
                '<input type="hidden" name="testProblemType" value="2">'+
                '<input type="hidden" name="testProblemTitle" id="'+bigTestTypeNum+"_title"+'">'+
                '<input type="hidden" id="'+bigTestTypeNum+"_smallNum"+'">'+	'<input type="hidden" name="testProblemOrder" id="'+"testBigNumber"+bigTestTypeNum+'"'+'value="'+bigTestTypeNum+'">'+
                '<input type="hidden" id="'+bigTestTypeNum+"_smallScore"+'">'+
                "</div>"+
                '<div class="addOneTestType">'+
                '<span style="font-size:17px;color:black;margin-left:2%;margin-top:10px;">添加小题数目:&nbsp</span>'+
                '<input type="number" style="width:60px;margin-top:10px;" value="1" id="'+bigTestTypeNum+"_addSmallNum"+'" >'+
                '每小题分值:<input type="number" style="width:60px;margin-top:10px;" value="1" id="'+bigTestTypeNum+"_eachSmallScore"+'"'+'name="'+bigTestTypeNum+"_eachSmallScore"+'"'+">"+
                '<input type="button" value="添加" style="background:green;color:white;text-align:center;padding-top:5px;width:80px;height:32px;border-style:none;border-radius:3px;margin-left:3%;margin-top:5px;" id="'+"addSmallButtonFillIn"+bigTestTypeNum+'" class="addSmallButtonFillIn">'+
                "</div>";
            $(".writeTestDetail").append(strhtml);
        }

    })

    $("#submitTestInfoButton").click(function(){
        $("#publicTeacherId").val(teacherId);
        $.ajax({
            type:"GET",
            url:"submitTestBaseInfo",
            cache:false,
            dataType:"json",
            data:$("#testBaseInfoForm").serialize(),
            success:function(data){
                if(data.result=="ok")
                {
                    alert("提交成功");
                    window.location.reload();
                }
            },
            error:function(e){
                alert(e.responseText);
            }

        })
    })
    $("#submitTestDetailButton").click(function(){
       $.ajax({
            type:"GET",
            url:"submitTestDetailInfo",
            cache:false,
            dataType:"json",
            data:$("#testDetailForm").serialize(),
            success:function(data){
               alert("ok");
            },
            error:function(e){
                alert(e.responseText);
            }

        })
    })
})
$(document).on("click", ".EditTest",function(){
    var id1 = $(this).attr("id");
    var id2=id1.substr(12,5);
   // alert("testId:"+id2);
    $("#testIds").val(id2);
    $(".testEdit").show();
    $(".testWaitRough").hide();
})
$(document).on("click", ".gotofinish", function () {//查看 按钮
    arrys = [];//记录每道大题中小题数目
   var  teststr = "";
    arryTypes = [];
    trueAnswer = [];
    score = [];
    var id1 = $(this).attr("id");
    $("#testIds").val(id1);
    //alert(chapterArr.length);
    for (var i = 0; i < chapterArr.length; i++) {
        var flag = false;
        for (var j = 0; j < chapterArr[i].length; j++) {
            if (chapterArr[i][j].testId == id1) {
                teststr = chapterArr[i][j].testDetail;
                var jsonstr = teststr;
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
                                '<div><span style="font-size:17px;color:grey;">你的答案：</span><input type="text" class="answers" id="' + (i + 1) + "_" + (j + 1) + '"style="margin-left:10px;border-left-style:none;border-right-style:none;border-top-style:none;border-bottom-color:black;border-bottom-width:thin;background-color:rgb(251, 247, 247);"/></div>&nbsp&nbsp&nbsp' + '<img class="correctImg" id="img' + (i + 1) + "_" + (j + 1) + '"/>' +
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
    $("#gobackrough").show();
})
$(document).on("click", ".begintest", function () {
    $(".testentry").hide();
    $("#gobackrough").hide();
    $(".testread").show();
})
$(document).on("click",".addSmallButtonChoose",function(){
    var id1=$(this).attr("id");
    var id2=id1.substring(20,22);
    var hehehe1=$("#"+id2+"_smallNum1").text();
    var hehehe=parseInt(hehehe1);//小题数目
    var allscore=$("#"+id2+"_allScore1").text();
    var allscore1=parseInt($("#"+id2+"_addSmallNum").val())*parseInt($("#"+id2+"_eachSmallScore").val());
    var allscore3=parseInt(allscore);
    var allscore2=allscore1+allscore3;
    $("#"+id2+"_smallScore1").text($("#"+id2+"_eachSmallScore").val());
    $("#"+id2+"_smallScore").val($("#"+id2+"_eachSmallScore").val());
    $("#"+id2+"_eachSmallScore").css("display","none");
    $("#"+id2+"_allScore").val(allscore2);
    $("#"+id2+"_allScore1").text(allscore2);
    for(var i=0;i<parseInt($("#"+id2+"_addSmallNum").val());i++){
        hehehe++;
        var	str1='<div class="chooseModal"><div>('+hehehe+')<textarea cols="118" rows="3" id="'+id2+"_"+hehehe+"title"+ '"name="smallProblemTitle">选择题题目</textarea>'+"</div>"+
            '<div>A.<textarea cols="118" rows="2" id="'+id2+"_"+hehehe+'A" name="smallProblemADetail">A选项内容</textarea></div>'+
            '<div>B.<textarea cols="118" rows="2" id="'+id2+"_"+hehehe+'B" name="smallProblemBDetail">B选项内容</textarea></div>'+
            '<div>C.<textarea cols="118" rows="2" id="'+id2+"_"+hehehe+'C" name="smallProblemCDetail">C选项内容</textarea></div>'+
            '<div>D.<textarea cols="118" rows="2" id="'+id2+"_"+hehehe+'D" name="smallProblemDDetail">D选项内容</textarea></div>'+
            '<div>正确答案:<input type="text" id="trueAnswer'+id2+"_"+hehehe+'" name="smallProblemTrueAnswer"></div>'+
            "</div>"+
             '<input type="hidden" name="belongTestProblemId" value="'+id2+'">'+
            '<input type="hidden" name="smallProblemType" value="0">'+
            '<input type="hidden" name="smallProblemScore" value="'+$("#"+id2+"_smallScore").val()+'">'+
            '<input type="hidden" name="smallProblemTip" value="null">';
        $("#"+id2+"_choose").append(str1);
    }
    $("#"+id2+"_smallNum1").text(hehehe);
    $("#"+id2+"_smallNum").val(hehehe);
    var str="选择题，共";
    var str1=str+hehehe+"道小题,每道题"+parseInt($("#"+id2+"_eachSmallScore").val())+"分，总共"+allscore2+"分";
    $("#"+id2+"_title").val(str1);
});
$(document).on("click",".addSmallButtonJudge",function(){
    var id1=$(this).attr("id");
    var id2=id1.substring(19,21);
    var hehehe1=$("#"+id2+"_smallNum1").text();
    var hehehe=parseInt(hehehe1);
    var allscore=$("#"+id2+"_allScore1").text();
    var allscore1=parseInt($("#"+id2+"_addSmallNum").val())*parseInt($("#"+id2+"_eachSmallScore").val());
    var allscore3=parseInt(allscore);
    var allscore2=allscore1+allscore3;
    $("#"+id2+"_smallScore1").text($("#"+id2+"_eachSmallScore").val());
    $("#"+id2+"_smallScore").val($("#"+id2+"_eachSmallScore").val());
    $("#"+id2+"_eachSmallScore").css("display","none");
    $("#"+id2+"_allScore").val(allscore2);
    $("#"+id2+"_allScore1").text(allscore2);
    for(var i=0;i<parseInt($("#"+id2+"_addSmallNum").val());i++){
        hehehe++;
        var	str1='<div class="judgeModal"><div>('+hehehe+')<textarea cols="118" rows="3" id="'+id2+"_"+hehehe+"title"+ '"name="smallProblemTitle">判断题题目</textarea>'+"</div>"+
            '<div >正确答案:<select name="smallProblemTrueAnswer"><option selected value="true">对</option><option value="false">错</option></select>'+
            "</div>"+
            "</div>"+
            "</div>"+
            '<input type="hidden" name="belongTestProblemId" value="'+id2+'">'+
            '<input type="hidden" name="smallProblemADetail" value="null">'+
            '<input type="hidden" name="smallProblemBDetail" value="null">'+
            '<input type="hidden" name="smallProblemCDetail" value="null">'+
            '<input type="hidden" name="smallProblemDDetail" value="null">'+
            '<input type="hidden" name="smallProblemType" value="1">'+
            '<input type="hidden" name="smallProblemScore" value="'+$("#"+id2+"_eachSmallScore").val()+'">'+
            '<input type="hidden" name="smallProblemTip" value="null">';
        $("#"+id2+"_judge").append(str1);
    }
    $("#"+id2+"_smallNum1").text(hehehe);
    $("#"+id2+"_smallNum").val(hehehe);
    var str="判断题，共";
    var str1=str+hehehe+"道小题,每道题"+parseInt($("#"+id2+"_eachSmallScore").val())+"分，总共"+allscore2+"分";
    $("#"+id2+"_title").val(str1);
});
$(document).on("click",".addSmallButtonFillIn",function(){
    var id1=$(this).attr("id");
    var id2=id1.substring(20,22);
    var hehehe1=$("#"+id2+"_smallNum1").text();
    var hehehe=parseInt(hehehe1);
    var allscore=$("#"+id2+"_allScore1").text();
    var allscore1=parseInt($("#"+id2+"_addSmallNum").val())*parseInt($("#"+id2+"_eachSmallScore").val());
    var allscore3=parseInt(allscore);
    var allscore2=allscore1+allscore3;
    $("#"+id2+"_smallScore1").text($("#"+id2+"_eachSmallScore").val());
    $("#"+id2+"_smallScore").val($("#"+id2+"_eachSmallScore").val());
    $("#"+id2+"_eachSmallScore").css("display","none");
    $("#"+id2+"_allScore").val(allscore2);
    $("#"+id2+"_allScore1").text(allscore2);
    for(var i=0;i<parseInt($("#"+id2+"_addSmallNum").val());i++){
        hehehe++;
        var	str1='<div class="fillinModal"><div>('+hehehe+')<textarea cols="118" rows="3" id="'+id2+"_"+hehehe+"title"+ '"name="smallProblemTitle">填空题题目</textarea>'+"</div>"+
            '<div >正确答案:<input type="text" id="'+id2+"_"+hehehe+"trueanswer"+ '"  name="smallProblemTrueAnswer">'+
            '指定答案类型<select id="'+id2+"_"+hehehe+"answerType"+ '" name="smallProblemTip">'+
            '<option value="number" selected>数字</option>'+
            '<option value="profeword" >专有名词</option>'+
            "</select>"+
            "</div>"+
            "</div>"+
            '<input type="hidden" name="belongTestProblemId" value="'+id2+'">'+
            '<input type="hidden" name="smallProblemType" value="2">'+
            '<input type="hidden" name="smallProblemADetail" value="null">'+
            '<input type="hidden" name="smallProblemBDetail" value="null">'+
            '<input type="hidden" name="smallProblemCDetail" value="null">'+
            '<input type="hidden" name="smallProblemDDetail" value="null">'+
            '<input type="hidden" name="smallProblemScore" value="'+$("#"+id2+"_eachSmallScore").val()+'">';
        $("#"+id2+"_fillin").append(str1);
    }
    $("#"+id2+"_smallNum1").text(hehehe);
    $("#"+id2+"_smallNum").val(hehehe);
    var str="填空题，共";
    var str1=str+hehehe+"道小题,每道题"+parseInt($("#"+id2+"_eachSmallScore").val())+"分，总共"+allscore2+"分";
    $("#"+id2+"_title").val(str1);
});
