<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/4/27 0027
  Time: 20:28
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.xqq.pojo.Teacher" %>
<% Teacher teacher=(Teacher)session.getAttribute("teacherInfo");
    Integer teacherId=-1;
    String  teacherPic="";
    String teacherName="";
    Integer belongSchId=-1;
    String teacherPhone="";
    //String teacherPassword="";
   // String teacherSex="";
   // String teacherAccount="";
   // String teacherIntroduce="";
    //String createDate="";
    //String teacherEmail="";
    if(teacher!=null)
    {
        teacherPhone=teacher.getTeacherPhone();
        teacherId=teacher.getTeacherId();
        teacherName=teacher.getTeacherName();
        teacherPic=teacher.getTeacherPic();
        //teacherPassword=teacher.getTeacherPassword();
        //teacherAccount=teacher.getTeacherAccount();
       // teacherIntroduce=teacher.getTeacherIntroduce();
       // teacherSex=teacher.getTeacherSex();
       // createDate=teacher.getCreateDate();
       // teacherEmail=teacher.getEmail();
        belongSchId=Integer.parseInt(teacher.getBelongSchId());
    }
    //System.out.println("nullStudent");
%>
<html>
<head >
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>教师主页</title>
    <link href="css/teacherownindex.css" rel="stylesheet" type="text/css">
    <link rel="stylesheet" href="css/bootstrap.min.css">
    <script src="js/jquery-2.0.3.js"></script>
    <script type="text/javascript" src="js/bootstrap.min.js"></script>
    <script>
        var teacherId=<%=teacherId%>;
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
            var dos=getUrlParam(name);
            if(dos=="course"){
                var time1=new Date().getTime();
                $("#111").css("color","orange");
                $.ajax({
                    type:"GET",
                    url:"teacherCourse?time="+time1,
                    dataType:"json",
                    cache:false,
                    success:function(data){
                        var counts=0;
                        var strhtml1="";
                        $(".shiftdiv").empty();
                        $("#countofmaster").html(data.length);
                        $("#allcourse").html(data.length);
                        $("#alreadycourse").html(data.length);
                        $.each(data,function(i,values){
                            counts++;
                            var strhtml1ss=
                                '<div class="coursedetail" id="course'+counts+
                                '">'+
                                '<div class="coursepic"><img src='+
                                '"./img/'+values["courseImgSrc"]+'"'+'style="height:100%;border-style:none;"></div>'+
                                '<div class="rightofcoursepic">'+
                                '<div class="coursename">'+
                                '<h3 style="color:black"><strong>'+values["courseName"]+"</strong></h3>"+
                                '<span style="background-color:rgba(0,0,0,0.4);color:white;fon-size:17px;"'+ 'id="courseflag'+counts+'">'+values["courseState"]+"</span></div>"+
                                '<div class="coursedisplay">'+
                                '<div class="coursebar" id="coursebar'+values["courseId"]+
                                '">'+
                                '<div id="myprogresses'+values["courseId"]+'"'+ 'style="height:inherit;background:red;width:1px;"></div></div>'+
                                '<div class="courseprogress"><span>'+values["Progress"]+"</span>/<span>"+values["chapterNum"]+"</span></div>"+
                                '<div class="courseDirectoryEntry" id="courseDirectoryEntry'+values["courseId"]+'"><a href="#a">进入管理</a></div></div></div></div>';
                            strhtml1+=strhtml1ss;
                            $(".shiftdiv").append(strhtml1ss);
                            //alert(strhtml1ss);
                            var myprogress=values["Progress"];
                            var progressall=values["chapterNum"];
                            var feiwusb=parseFloat(myprogress)/parseFloat(progressall);
                            var widths=$(".coursebar").width();
                            var myprogresswidth=parseFloat(widths)*feiwusb;
                            var myprogresswidthint=parseInt(myprogresswidth);
                            //alert(strhtml1);
                            $("#myprogresses"+values["courseId"]).css("width",myprogresswidthint+"px");
                            //alert("进度条总宽度为:"+widths+"比例为:"+feiwusb+"我的进度的宽度为"+myprogresswidthint+"实际我的进度条宽度为"+$("#myprogresses"+values["courseId"]).width());
                        })
                        $(".coursehead").show();
                        $(".coursedetail").show();
                        $(".nonecourse").hide();


                    },
                    error:function(xhr){
                        alert(xhr.responseText);
                        $(".shiftdiv").empty();
                        $(".coursehead").hide();
                        $(".coursedetail").hide();
                        $(".nonecourse").show();
                    }

                })
                $(".mycourseinfo").css("display","block");
                $(".myexcerciseinfo").css("display","none");
                $(".myscoreinfo").css("display","none");
                $(".mycapacityinfo").css("display","none");
                $("#managecourse").css("color","orange");

            }
            else if(dos=="excercise")
            {
                $(".mycourseinfo").css("display","none");
                $(".myexcerciseinfo").css("display","block");
                $(".myscoreinfo").css("display","none");
                $(".mycapacityinfo").css("display","none");
                $("#excercises").css("color","orange");
            }
            else if(dos=="datas"){
                $(".mycourseinfo").hide();
                $(".myexcerciseinfo").hide();
                $(".myscoreinfo").show();
                $(".mycapacityinfo").hide();
                $("#anlysisdata").css("color","orange");
                $.ajax({
                    type:"GET",
                    url:"scoreInfo",
                    data:{"teacherId":teacherId},
                    dataType:"json",
                    cache:false,
                    success:function(data){
                        $(".myscoreinfo").empty();
                        $.each(data,function(i,value){
                            var strhtm='<div class="oneCourseScore" style="margin-top:5%;" id="oneCourseScore'+i+'">\n' +
                                '            <div><strong><a href="courseIntroduce.jsp?courseid='+value.courseId+'">'+value.courseName+'</a></strong></div>\n' +
                                '            <div id="oneScoreDetail'+i+'"style="margin-left:2%;">\n' +
                                '            </div>\n' +
                                '        </div><hr>';
                            $(".myscoreinfo").append(strhtm);
                            //alert("testOrExamList.length:"+value.testOrExamList.length);
                            $.each(value.testOrExamList,function(j,vals){
                                var strhtm1='<div class="testOrExam" style="text-align: center;margin-top:2%;">'+vals.testOrExamName+'</div>\n' +
                                    '                <div id="scoreTableDiv'+i+j+'">\n' +
                                    '                    <table id="scoreTable'+i+j+'" style="text-align: center;">\n' +
                                    '                        <tr>\n' +
                                    '                            <td>\n' +
                                    '                                姓名\n' +
                                    '                            </td>\n' +
                                    '                            <td>\n' +
                                    '                                学号\n' +
                                    '                            </td>\n' +
                                    '                            <td>\n' +
                                    '                                成绩\n' +
                                    '                            </td>\n' +
                                    '                        </tr>\n' +
                                    '                    </table>\n' +
                                    '                    <div style="text-align: center;">及格率:<span class="passPerc1">'+vals.passPerc+'</span>&nbsp&nbsp&nbsp&nbsp优秀率:<span class="goodPerc1">'+vals.goodPerc+'</span></div>\n' +
                                    '                </div>';
                                $("#oneScoreDetail"+i).append(strhtm1);
                                $.each(vals.scoreList,function(k,values){
                                    var strhtm2=' <tr>\n' +
                                        '                            <td>\n' + values.scoreBelongName +
                                        '                            </td>\n' +
                                        '                            <td>\n' +values.scoreBelongAccount+
                                        '                            </td>\n';
                                    if(values.scoreMark>=80)
                                    {
                                        var strhtm3='<td style="color:green;">'+values.scoreMark+'</td>';
                                        strhtm2+=strhtm3;
                                    }
                                    else if(values.scoreMark>=60)
                                    {
                                        var strhtm3='<td style="color:yellow;">'+values.scoreMark+'</td>';
                                        strhtm2+=strhtm3;
                                    }
                                    else
                                    {
                                        var strhtm3='<td style="color:red;">'+values.scoreMark+'</td>';
                                        strhtm2+=strhtm3;
                                    }
                                       strhtm2+= '</tr>';
                                    //alert(strhtm2);
                                    $("#scoreTable"+i+j).append(strhtm2);
                                })

                            })
                            var strhtm3='<div class="Exam" style="color:indianred;text-align:center;margin-top:3%;">'+value.courseName+"总成绩"+'</div>\n' +
                                '                <div id="examTableDiv'+i+'">\n' +
                                '                    <table id="examTable'+i+'" class="table">\n' +
                                    '<thead>'+
                                '                        <tr>\n' +
                                '                            <td style="width:13%;">\n' +
                                '                                姓名\n' +
                                '                            </td>\n' +
                                '                            <td style="width:13%;">\n' +
                                '                                学号\n' +
                                '                            </td>\n' +
                                '                            <td style="width:13%;">\n' +
                                '                                论坛成绩（100）\n' +
                                '                            </td>\n' +
                                '                            <td style="width:13%;">\n' +
                                '                                平时作业成绩（100）\n' +
                                '                            </td>\n' +
                                '                            <td style="width:13%;">\n' +
                                '                                期末测试成绩（100）\n' +
                                '                            </td>\n' +
                                '                            <td style="width:13%;">\n' +
                                '                                总成绩（100）\n' +
                                '                            </td>\n' +
                                '                            <td style="width:13%;">\n' +
                                '                                证书\n' +
                                '                            </td>\n' +
                                '                        </tr>\n' +
                                    '</thead>'+'<tbody id="examTBody'+i+'"></tbody>'+
                                '                    </table>\n' +
                                '                    <div style="text-align: center;">及格率:<span class="passPerc1">'+value.passPerc+'</span>&nbsp&nbsp&nbsp&nbsp优秀率:<span class="goodPerc1">'+value.goodPerc+'</span></div>\n' +
                                '                </div>';
                            $("#oneCourseScore"+i).append(strhtm3);
                            $.each(value.examList,function(n,values){
                                var st='<tr class="info">' +
                                    '    <td style="width:13%;">' +
                                    values.studentName +
                                    ' </td>' +
                                    '<td style="width:13%;">' +
                                    values.studentAccount +
                                    ' </td>' +
                                    ' <td style="width:13%;">' +
                                    values.commentScore +
                                    '</td>' +
                                    ' <td style="width:13%;">' +
                                    values.testScore +
                                    '</td>' +
                                    ' <td style="width:13%;">' +
                                    values.examScore +
                                    '</td>' +
                                    ' <td style="width:13%;">' +
                                    values.allScore +
                                    '</td>';
                                if(values.allScore<60){
                                    st+='<td style="color:red;width:13%;">' +
                                            "不通过"+
                                        '</td></tr>'
                                }else if(values.allScore<80){
                                    st+='<td style="color:orange;width:13%;">' +
                                        "良好"+
                                        '</td></tr>'
                                }else{
                                    st+='<td style="color:green;width:13%;">' +
                                        "优秀"+
                                        '</td></tr>'
                                }
                                $("#examTBody"+i).append(st);
                            })
                        })
                    },
                    error:function(xhr){
                        alert(xhr.responseText);
                    }

                })
            }
            else if(dos=="courseAppl"){
                $(".mycourseinfo").css("display","none");
                $(".myexcerciseinfo").css("display","none");
                $(".myscoreinfo").css("display","none");
                $(".mycapacityinfo").css("display","block");
                $("#capacities").css("color","orange");
            }
        }
        $(function(){
            var time1=new Date().getTime();
            $.ajax({
                type:"GET",
                url:"teacherInfo?time="+time1,
                data:{"teacherId":teacherId},
                dataType:"json",
                cache:false,
                success:function(data){
                    //alert(data.belongSchName);
                    $("#alreadycourse").html(data.courseNum);
                    $("#belongSchName").html(data.belongSchName);
                },
                error:function(xhr){
                    alert(xhr.responseText);
                }

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
                    case 111:$("#111").css("color","orange");$("#222").css("color","grey");$("#333").css("color","grey");$("#444").css("color","grey");
                        $(".nonecourse").css("display","none");
                        $(".coursedetail").show();
                        $("#countofmaster").html($("#allcourse").text());
                        break;
                    case 222:$("#222").css("color","orange");$("#111").css("color","grey");$("#333").css("color","grey");$("#444").css("color","grey");
                        var course="course";
                        var courseflag="courseflag";
                        var sb;
                        var coursecount=0;
                        var length=parseInt($("#allcourse").text());
                        var i=0;
                        for(i=1;i<=length;i++){
                            sb=courseflag+i;
                            if($("#"+sb).text()=="正在进行")
                            {
                                $("#"+course+i).css("display","block");
                                coursecount++;
                            }
                            else
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
                    case 333:$("#333").css("color","orange");$("#222").css("color","grey");$("#111").css("color","grey");$("#444").css("color","grey");
                        //$(".nonecourse").css("display","block");
                        //$(".coursedetail").css("display","none");
                        var course="course";
                        var courseflag="courseflag";
                        var sb;
                        var coursecount=0;
                        var length=parseInt($("#allcourse").text());
                        for(i=1;i<=length;i++){
                            sb=courseflag+i;
                            if($("#"+sb).text()=="已完结")
                            {
                                $("#"+course+i).css("display","block");
                                coursecount++;
                            }
                            else
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
                    case 444:$("#444").css("color","orange");$("#222").css("color","grey");$("#111").css("color","grey");$("#333").css("color","grey");
                        var course="course";
                        var courseflag="courseflag";
                        var sb;
                        var coursecount=0;
                        var length=parseInt($("#allcourse").text());
                        for(i=1;i<=length;i++){
                            sb=courseflag+i;
                            if($("#"+sb).text()=="未开课")
                            {
                                $("#"+course+i).css("display","block");
                                coursecount++;
                            }
                            else
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
            $("#type1").change(function(){
                var type1=$("#type1").val();
                //alert(type1);
                $("#type3").empty();
                if(type1=="文科")
                {
                    $("#type2").empty();
                    $("#type2").append('<option value="语言">语言</option>');
                    $("#type2").append('<option value="政治">政治</option>');
                    $("#type2").append('<option value="历史">历史</option>');
                    $("#type2").append('<option value="其他">其他</option>');
                    $("#type3").append('<option value="中文">中文</option>');
                    $("#type3").append('<option value="英语">英语</option>');
                    $("#type3").append('<option value="日语">日语</option>');
                    $("#type3").append('<option value="法语">法语</option>');
                    $("#type3").append('<option value="俄语">俄语</option>');
                    $("#type3").append('<option value="韩语">韩语</option>');
                }
                else if(type1=="理科")
                {
                    $("#type2").empty();
                    $("#type2").append('<option value="数学">数学</option>');
                    $("#type2").append('<option value="地理">地理</option>');
                    $("#type2").append('<option value="物理">物理</option>');
                    $("#type2").append('<option value="化学">化学</option>');
                    $("#type2").append('<option value="生物">生物</option>');
                    $("#type3").append('<option value="线性代数">线性代数</option>');
                    $("#type3").append('<option value="高等数学">高等数学</option>');
                    $("#type3").append('<option value="数学分析">数学分析</option>');
                    $("#type3").append('<option value="概率论">概率论</option>');
                }
                else {
                    $("#type2").empty();
                    $("#type2").append('<option value="计算机">计算机</option>');
                    $("#type2").append('<option value="机械">机械</option>');
                    $("#type2").append('<option value="电子信息">电子信息</option>');
                    $("#type3").append('<option value="软件开发">软件开发</option>');
                    $("#type3").append('<option value="编程语言">编程语言</option>');
                    $("#type3").append('<option value="嵌入式开发">嵌入式开发</option>');
                    $("#type3").append('<option value="算法理论">算法理论</option>');
                    $("#type3").append('<option value="机器学习">机器学习</option>');
                    $("#type3").append('<option value="数据挖掘">数据挖掘</option>');
                }
            })


            $("#type2").change(function(){
                var type2=$("#type2").val();
                if(type2=="语言")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="中文">中文</option>');
                    $("#type3").append('<option value="英语">英语</option>');
                    $("#type3").append('<option value="日语">日语</option>');
                    $("#type3").append('<option value="法语">法语</option>');
                    $("#type3").append('<option value="俄语">俄语</option>');
                    $("#type3").append('<option value="韩语">韩语</option>');
                }
                else if(type2=="政治")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="国际形势">国际形势</option>');
                    $("#type3").append('<option value="思想理论">思想理论</option>');
                    $("#type3").append('<option value="国内政治">国内政治</option>');
                }
                else if(type2=="历史")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="现代国内历史">现代国内历史</option>');
                    $("#type3").append('<option value="古代国内历史">古代国内历史</option>');
                    $("#type3").append('<option value="近现代国内历史">近现代国内历史</option>');
                    $("#type3").append('<option value="现代国外历史">现代国外历史</option>');
                    $("#type3").append('<option value="古代国外历史">古代国外历史</option>');
                    $("#type3").append('<option value="近现代国外历史">近现代国外历史</option>');
                }
                else if(type2=="其他")
                {
                    $("#type3").empty();
                }
                else if(type2=="数学")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="线性代数">线性代数</option>');
                    $("#type3").append('<option value="高等数学">高等数学</option>');
                    $("#type3").append('<option value="数学分析">数学分析</option>');
                    $("#type3").append('<option value="概率论">概率论</option>');
                }
                else if(type2=="地理")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="天文">天文</option>');
                    $("#type3").append('<option value="水土">水土</option>');
                    $("#type3").append('<option value="人文地理">人文地理</option>');
                }
                else if(type2=="物理")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="量子力学">量子力学</option>');
                    $("#type3").append('<option value="相对论">相对论</option>');
                    $("#type3").append('<option value="经典物理">经典物理</option>');
                }
                else if(type2=="化学")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="中学化学">中学化学</option>');
                    $("#type3").append('<option value="大学化学">大学化学</option>');

                }
                else if(type2=="生物")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="医学理论">医学理论</option>');
                    $("#type3").append('<option value="大学生物">大学生物</option>');
                    $("#type3").append('<option value="生物常识理论">生物常识理论</option>');
                }
                else if(type2=="计算机")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="软件开发">软件开发</option>');
                    $("#type3").append('<option value="编程语言">编程语言</option>');
                    $("#type3").append('<option value="嵌入式开发">嵌入式开发</option>');
                    $("#type3").append('<option value="算法理论">算法理论</option>');
                    $("#type3").append('<option value="机器学习">机器学习</option>');
                    $("#type3").append('<option value="数据挖掘">数据挖掘</option>');
                }
                else if(type2=="机械")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="机械理论">机械理论</option>');
                    $("#type3").append('<option value="自动化">自动化</option>');

                }
                else if(type2=="电子信息")
                {
                    $("#type3").empty();
                    $("#type3").append('<option value="电路基础">电路基础</option>');
                    $("#type3").append('<option value="电子开发">电子开发</option>');
                }

            })
            $("#lookAlrAppBtn").click(function(){
                $("#applicateCourseFormDiv").hide();
                $.ajax({
                    type:"GET",
                    url:"applicateCourseState",
                    data:{"teacherId":teacherId},
                    dataType:"json",
                    cache:false,
                    success:function(data){
                        $("#alreadyAppl").empty();
                        $("#alreadyAppl").append("已申请开课列表")
                       var htmlstr="";
                        $.each(data,function(i,value){
                            htmlstr="";
                            if(value.checkState=="申请失败")
                            {
                                htmlstr='<div class="oneAppl" style="width:100%;height:80px;border-width:1px;border-style:solid;border-color:greenyellow">\n' +
                                    '                <div>\n' +
                                    "                 <span ><strong>"+value.courseName+"</strong></span>\n" +
                                    '                    <span style="color:red;margin-left:5%;">申请失败</span>\n' +
                                    '                </div>\n' +
                                    '               <div>\n' +
                                    '                    <div style="float:left;"> 说明:</div>\n' +
                                    '                    <div class="suplementDiv" style="float:left;">\n' +value.suplement+
                                    '                    </div>\n' +
                                    '                </div>\n' +
                                    '                <div style="margin-left:50%;">\n' +
                                    '                    <input type="button" value="确认" id="checkFail'+value.checkId+'" class="checkFail">\n' +
                                    '                </div>\n' +
                                    '            </div>';
                            }
                            else if(value.checkState=="申请成功")
                            {
                                htmlstr='<div class="oneAppl" style="width:100%;height:80px;border-width:1px;border-style:solid;border-color:greenyellow">\n' +
                                    '                <div>\n' +
                                    "                   <span ><strong>"+value.courseName+"</strong></span>\n"+
                                    '                    <span style="color:green;margin-left:5%;">申请成功</span>\n' +
                                    '                </div>\n' +
                                    '                <div>\n' +
                                    '                    <div style="float:left;"> 说明:</div>\n' +
                                    '                    <div class="suplementDiv" style="float:left;">\n' +value.suplement+
                                    '                    </div>\n' +
                                    '                </div>\n' +
                                    '                <div style="margin-left:50%;">\n' +
                                    '                    <input type="button" value="确认" id="checkSuccess'+value.checkId+'" class="checkSuccess">\n' +
                                    '                </div>\n' +
                                    '            </div>';
                            }
                            else if(value.checkState=="待审核")
                            {
                                htmlstr='<div class="oneAppl" style="width:100%;height:30px;border-width:1px;border-style:solid;border-color:greenyellow">\n' +
                                    "              <div>\n" +
                                    "                   <span><strong>"+value.courseName+"</strong></span>\n" +
                                    '                    <span style="color:greenyellow;margin-left:5%;">待审核</span>\n' +
                                    "                </div>\n" +
                                    "          </div>";
                            }
                            $("#alreadyAppl").append(htmlstr);
                        })
                    },
                    error:function(xhr){
                        alert(xhr.responseText);
                    }

                })
                $("#alreadyAppl").show();
                $("#applicateCourseUtilAndChapterDiv").hide();
            })
            $("#ApplFormBtn").click(function(){
                $("#applicateCourseFormDiv").show();
                $("#alreadyAppl").hide();
                $("#applicateCourseUtilAndChapterDiv").hide();
            })
            $("#submitAppliCourseBtn").click(function(){
                var type1=$("#type1").val();
                var type2=$("#type2").val();
                var type3=$("#type3").val();
                var type="";
                if(type3==null||type3=="")
                    type=type1+"|"+type2;
                else
                   type=type1+"|"+type2+"|"+type3;
               // alert(type);
                $("#courseType1").val(type);
            })
            $("#checkAddChapters").click(function(){
                var chapterNum=$("#chapterNum").val();
                var htmstr="";
                var checkId=$("#hiddenBelongCourseId").val();
                $(".chapters").empty();
                for(var i=0;i<chapterNum;i++)
                {
                    htmstr=' <div class="oneChapter" id="chapter'+i+'" style="border-width:2px;border-style:solid;border-color:green;margin-top:3%;">\n' +
                        '                    <input type="hidden" name="belongCourseId" value="'+checkId+'">\n' +
                        '                    单元标题:<input type="text" name="chapterTitle" placeholder="单元标题，如：第一单元"><br>\n' +
                        '                    单元名字:<input type="text" name="chapterName" placeholder="单元名字，如：welcome to England"><br>\n' +
                        '                    单元顺序:<input type="number" name="chapterOrder" value="'+(i+1)+'" ><br>\n' +
                        '                    <input type="button" value="添加小节" class="addUtilBtn" id="addUtilBtn'+i+'">&nbsp添加小节数:<input type="number" placeholder="0" id="addUtilNum'+i+'">\n' +
                        '                    <div class="utils" id="util'+i+'">\n' +
                        '                    </div>\n' +
                        '                </div>';
                    $(".chapters").append(htmstr);
                }

            })
            $(document).on("click", ".addUtilBtn", function () {
                var thisid = $(this).attr("id");
                var order = thisid.substr(10, 3);
                var utilNum=$("#addUtilNum"+order).val();
                var htmstr="";
                $("#util"+order).empty();
                for(var i=0;i<utilNum;i++)
                {
                    htmstr='<div class="oneUtil" id="util'+order+i+'" style="border-width:1px;border-style:solid;border-color:orange;margin-top:1%">\n' +
                        '                            <input type="hidden" name="belongChapterId" value="'+order+'">\n' +
                        '                            小节标题:<input type="text" name="utilTitle" placeholder="小节标题，如：第一小节"><br>\n' +
                        '                            小节名字:<input type="text" name="utilName" placeholder="小节名字，如：welcome to England"><br>\n' +
                        '                            小节顺序:<input type="number" name="utilOrder" value="'+(i+1)+'" ><br>\n' +
                        '                        </div>';
                    $("#util"+order).append(htmstr);
                }
            })

           $(document).on('click',".checkSuccess",function(){
               $(".chapters").empty();
               var id=$(this).attr("id");
               var belongCourseId=id.substr(12,5);
               $("#hiddenBelongCourseId").val(belongCourseId);
              // alert(belongCourseId);
               $("#applicateCourseFormDiv").hide();
               $("#alreadyAppl").hide();
               $("#applicateCourseUtilAndChapterDiv").show();
           })
        })
        $(document).on("click", ".courseDirectoryEntry", function () {
            var thisid = $(this).attr("id");
            var courseid = thisid.substr(20, 3);
            window.location.href = "courseIntroduce.jsp?courseid=" + courseid;
        })
    </script>
    <style>
        table{
            width:95%;
            margin-left:2%;
            text-align: center;
        }
        tr{
            border-style:solid;
            border-width:1px;
            border-color:antiquewhite;
            width:100%;
            height:30px;
        }
        td{
            border-style:solid;
            border-width:1px;
            border-color:antiquewhite;
            text-align: center;
            height:inherit;
        }
    </style>
</head>
<body onLoad="al('do')">
<span style="display:none;" id="allcourse"></span>
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
                <img  style=" float:left;height:180px;margin-left:200px;width:180px;border-radius:90px;border-style:none;" src="<%="./img/"+teacherPic %>">
                <span style="float:left;margin-top:85px;width:auto;margin-left:10px;"><a style="color:white;font-size:25px;text-decoration: none;"><strong><%=teacherName %></strong></a></span>
            </div>
            <div class="rightaccount">
                <div class="schoolinfo"><div>来自:<span style="color:white;font-size:17px;" id="belongSchName"></span></div>
                </div>
                <div style="float:left;margin-left:57px;margin-top:115px;">
                    <div id="alreadycourse"><strong>0</strong></div>
                    <div>已管理课程</div>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="menu">
    <ul class="menus">
        <li style="margin-left:100px;"><div style="width:100px;height:inherit;font-size:17px;"><a href="teachers.jsp?do=course"><strong id="managecourse">课程管理</strong></a></div></li>
        <li><div style="width:100px;height:inherit;font-size:17px;"><a href="teachers.jsp?do=excercise"><strong id="excercises">综合实训</strong></a></div></li>
        <li><div style="width:100px;height:inherit;font-size:17px;"><a href="teachers.jsp?do=datas"><strong id="anlysisdata">数据分析</strong></a></div></li>
        <li><div style="width:100px;height:inherit;font-size:17px;"><a href="teachers.jsp?do=courseAppl"><strong id="courseAppl">申请开课</strong></a></div></li>
    </ul>
</div>
<div class="detail">
    <!--我的课程部分开始-->
    <div class="mycourseinfo">
        <div class="coursehead">
            <div style="float:left;width:10%;color:grey;font-size:19px;margin-top:20px;height:50%;margin-left:251px;">共<span id="countofmaster">0</span>门课程</div>
            <ul style="float:left;width:30%;margin-left:30%;color:grey;height:50%;margin-top:20px;" id="coursetype">
                <li id="111">全部</li>
                <li id="222">正在进行</li>
                <li id="333">已结束</li>
                <li id="444">未开课</li>
            </ul>
        </div>
        <div class="shiftdiv">
            <div class="nonecourse">
                暂无课程
            </div>

        </div>
    </div>

    <!--我的课程结束-->

    <!--综合实训开始-->
    <div class="myexcerciseinfo">

    </div>
    <!--综合实训结束-->

    <!--成绩管理开始-->
    <div class="myscoreinfo">
        <div class="oneCourseScore">
            <div id="courseName"><strong>计算机组成原理</strong></div>
            <div class="oneScoreDetail" style="margin-left:2%;">
                <div class="scoreName">第一单元测试</div>
                <div class="scoreTableDiv">
                    <table id="scoreTable" style="text-align: center;">
                        <tr>
                            <td>
                                姓名
                            </td>
                            <td>
                                学号
                            </td>
                            <td>
                                成绩
                            </td>
                        </tr>
                        <tr>
                            <td>
                                王宝强
                            </td>
                            <td>
                                84894789
                            </td>
                            <td>
                                87
                            </td>
                        </tr>
                        <tr>
                            <td>
                                王大锤
                            </td>
                            <td>
                                48979885
                            </td>
                            <td>
                                59.5
                            </td>
                        </tr>
                    </table>
                    <div style="text-align: center;">及格率:<span id="passPerc">50%</span>&nbsp&nbsp&nbsp&nbsp优秀率:<span id="goodPerc">20%</span></div>
                </div>
            </div>
        </div>
        <div class="oneCourseScore">
            <div id="courseName1"><strong>计算机组成原理</strong></div>
            <div class="oneScoreDetail" style="margin-left:2%;">
                <div class="scoreName">第一单元测试</div>
                <div class="scoreTableDiv">
                    <table id="scoreTable1" style="text-align: center;">
                        <tr>
                            <td>
                                姓名
                            </td>
                            <td>
                                学号
                            </td>
                            <td>
                                成绩
                            </td>
                        </tr>
                        <tr>
                            <td>
                                王宝强
                            </td>
                            <td>
                                84894789
                            </td>
                            <td>
                                <span style="color:green;"> 87</span>
                            </td>
                        </tr>
                        <tr>
                            <td>
                                王大锤
                            </td>
                            <td>
                                48979885
                            </td>
                            <td>
                                <span style="color:red;"> 59.5</span>
                            </td>
                        </tr>
                    </table>
                    <div style="text-align: center;">及格率:<span id="passPerc1">50%</span>&nbsp&nbsp&nbsp&nbsp优秀率:<span id="goodPerc1">20%</span></div>
                </div>
            </div>
        </div>
    </div>
    <!--我的成绩结束-->

    <!--申请开课开始-->
    <div class="mycapacityinfo">
        <div>
            <input type="button" value="查看已申请列表" id="lookAlrAppBtn">
            <input type="button" value="课程申请" id="ApplFormBtn">
        </div>
        <div id="applicateCourseFormDiv">
            <form style="margin-left:25%;margin-right:25%;margin-top:3%;" action="./applicateCourse" method="post" enctype="multipart/form-data" id="applicateCoruseForm" target="hidden_iframe1">
                课程名称:<input type="text" name="courseName" placeholder="课程名称"><br>
                 <input type="hidden" name="teacherId" value="<%=teacherId%>">
                <input type="hidden" name="belongSchId" value="<%=belongSchId%>">
                课程介绍:
                <textarea cols="90" rows="4" name="courseIntroduce" placeholder="课程介绍"></textarea>
                <br>
                上传课程封面:<input type="file" name="posterSrc"><br>
                上传简介视频:<input type="file" name="introduceSrc">
                设置课程类型:
                <select id="type1">
                    <option value="文科">文科</option>
                    <option value="理科">理科</option>
                    <option value="工科">工科</option>
                </select>
                <select id="type2">
                    <option value="语言">语言</option>
                    <option value="政治">政治</option>
                    <option value="历史">历史</option>
                    <option value="其他">其他</option>
                </select>
                <select id="type3">
                    <option value="中文">中文</option>
                    <option value="英语">英语</option>
                    <option value="日语">日语</option>
                    <option value="法语">法语</option>
                    <option value="俄语">俄语</option>
                    <option value="韩语">韩语</option>
                </select>
                <input type="hidden" name="courseType" id="courseType1">
                <input type="hidden" name="teacherPhone" value="<%=teacherPhone%>">
                <br>
                <input type="submit" id="submitAppliCourseBtn" value="提交">
                <iframe name="hidden_iframe1" style="display:none;"></iframe>
            </form>
        </div>
        <div id="alreadyAppl" style="width:60%;margin-left:20%;height:auto;display:none;">
            已申请开课列表
           <!-- <div class="oneAppl" style="width:100%;height:80px;border-width:1px;border-style:solid;border-color:greenyellow">
                <div>
                    <span id="applCourseName"><strong>探索发现</strong></span>
                    <span style="color:red;margin-left:5%;">申请失败</span>
                </div>
               <div>
                    <div style="float:left;"> 说明:</div>
                    <div class="suplementDiv" style="float:left;">
                    没有教师资格证
                    </div>
                </div>
                <div style="margin-left:50%;">
                    <input type="button" value="确认" id="checkFail" class="checkFail">
                </div>
            </div>
            <div class="oneAppl" style="width:100%;height:80px;border-width:1px;border-style:solid;border-color:greenyellow">
                <div>
                    <span id="applCourseName1"><strong>探索发现</strong></span>
                    <span style="color:green;margin-left:5%;">申请成功</span>
                </div>
                <div>
                    <div style="float:left;"> 说明:</div>
                    <div class="suplementDiv" style="float:left;">
                        请尽快为本课程添加章节，并到该课程管理界面管理
                    </div>
                </div>
               <!-- <div style="margin-left:50%;">
                    <input type="button" value="确认" id="checkSuccess" class="checkSuccess">
                </div>
            </div>

            <div class="oneAppl" style="width:100%;height:30px;border-width:1px;border-style:solid;border-color:greenyellow">
                <div>
                    <span id="applCourseName3"><strong>探索发现</strong></span>
                    <span style="color:greenyellow;margin-left:5%;">待审核</span>
                </div>
            </div>-->
        </div>


        <div id="applicateCourseUtilAndChapterDiv" style="display:none;">
            <input type="hidden" id="hiddenBelongCourseId">
            <form action="addChapterAndUtil" method="post" id="addChapterAndUtilForm" target="hidden_iframe2">
           设置总章数: <input type="number" name="chapterNum" id="chapterNum" placeholder="0" min="1" max="12">&nbsp
            <input type="button" value="确定" id="checkAddChapters">
            <div class="chapters">
                <div class="oneChapter" id="chapter1" style="border-width:1px;border-style:solid;border-color:green;margin-top:3%;">
                    <input type="hidden" name="belongCourseId">
                    单元标题:<input type="text" name="chapterTitle" placeholder="单元标题，如：第一单元"><br>
                    单元名字:<input type="text" name="chapterName" placeholder="单元名字，如：welcome to England"><br>
                    单元顺序:<input type="number" name="chapterOrder" ><br>
                    <input type="button" value="添加小节" class="addUtilBtn" id="addUtilBtn1">&nbsp添加小节数:<input type="number"  id="addUtilNum1">
                    <div class="utils" id="util1">
                    </div>
                </div>
                <div class="oneChapter" id="chapter2" style="border-width:1px;border-style:solid;border-color:green;margin-top:3%;">
                    <input type="hidden" name="belongCourseId">
                    单元标题:<input type="text" name="chapterTitle" placeholder="单元标题，如：第一单元"><br>
                    单元名字:<input type="text" name="chapterName" placeholder="单元名字，如：welcome to England"><br>
                    单元顺序:<input type="number" name="chapterOrder" ><br>
                    <input type="button" value="添加小节" class="addUtilBtn" id="addUtilBtn2">&nbsp添加小节数:<input type="number" max="15" min="1" id="addUtilNum2">
                    <div class="utils" id="util2">
                        <div class="oneUtil" id="util21" style="border-width:1px;border-style:solid;border-color:orange;margin-top:1%">
                            <input type="hidden" name="belongChapterId" value="">
                            小节标题:<input type="text" name="utilTitle" placeholder="小节标题，如：第一小节"><br>
                            小节名字:<input type="text" name="utilName" placeholder="小节名字，如：welcome to England"><br>
                            小节顺序:<input type="number" name="utilOrder" ><br>
                        </div>
                        <div class="oneUtil" id="util22" style="border-width:1px;border-style:solid;border-color:orange;margin-top:1%">
                            <input type="hidden" name="belongChapterId" value="">
                            小节标题:<input type="text" name="utilTitle" placeholder="小节标题，如：第一小节"><br>
                            小节名字:<input type="text" name="utilName" placeholder="小节名字，如：welcome to England"><br>
                            小节顺序:<input type="number" name="utilOrder" ><br>
                        </div>
                    </div>
                </div>
            </div>
                <input type="submit" value="提交">
                <iframe name="hidden_iframe2" style="display:none;"></iframe>
            </form>
        </div><!--为课程添加章节-->
    </div>
    <!--申请开课结束-->
</div>

</body>
</html>
