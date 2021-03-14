<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2019/4/14 0014
  Time: 13:54
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
<style>
    #studentExamTable  tr {
        cursor: pointer;
        background-color:burlywood;
    }
    #studentExamTable  tr td:hover{
        cursor: pointer;
        background-color:cadetblue;
    }
</style>
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
<style>
    .vjs-default-skin .vjs-big-play-button {left:40%;top:40%;}
    div{
        white-space:nowrap;
    }
</style>
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
            <li><a href="adminLogin.jsp">网站管理</a></li>
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
                <h3><strong style="color:black;">亲爱的<%=teacherName%>老师</strong></h3>
                <h4 id="welcomeName"></h4>
            </div>
            <input type="button" value="新建公告" style="width:20%;margin-left:110px;margin-top:30px;height:35px;text-align:center;padding-top:6px;color:white;background:rgba(175,130,18,0.88);float:left;border-radius: 4px;border-style:none;" id="addnotice">
        </div>
        <div class="nonenotice"><strong>暂时没有公告</strong></div>
        <div id="noticeview" style="height:auto;width:100%;">
            <div class="noti"><span style="margin-left:20px;" id="alreadypro">已发布公告</span><span style="margin-left:180px;" id="waitpro">待发布公告</span></div>
            <hr>
            <div class="noticedetail">
                <div style="height:auto;width:100%;" id="alreadyprodiv"><!--已发布公告-->
                </div>
                <div style="height:auto;width:100%;display: none;" id="waitprodiv">

                </div><!--待发布公告-->
            </div>
    </div>
        <div id="addnoticeview" style="height:auto;width:100%;display:none;">
            <img src="./img/t01ef39187ba1e08a27.png" style="height:23px;margin-left:251px;"><span id="gobacknoticeview">返回</span>
            <div class="addnoticeoption">
                <form action="" method="post" id="addNoticeForm">
                    <div>请输入公告标题:<input type="text" placeholder="标题" name="noticeTitle" id="noticetitle" style="width:50%;height:auto;color:black;font-size:22px;"></div>
                    <div >请输入公告具体内容:</div>
                    <div ><textarea cols="83" rows="13" style="color:grey;background:rgba(242,248,228,1.00);font-size:17px;border-style: solid;" id="noticebody" name="noticeDetail"></textarea></div>
                    <div >请输入公告发布人姓名:<input type="text" placeholder="姓名" name="writer"><span style="margin-left:100px;">请输入公告发布时间:</span><input type="date" placeholder="日期" name="time" id="noticetime"></div>
                    <input type="hidden" name="belongCourseId" id="belongCourseId">
                    <div >
                        <input type="button" value="修改" id="updateaddnotice" style="background:grey;color:white;text-align:center;width:95px;border-radius:2px;height:27px;border-style:none;" disabled>
                        <input type="button" value="保存" id="saveaddnotice"  style="background:green;color:white;text-align:center;width:95px;border-radius:2px;height:27px;border-style:none;">
                        <input type="button" value="发布" id="proaddnotice" style="background:grey;color:white;text-align:center;width:95px;border-radius:2px;height:27px;border-style:none;" disabled>
                    </div>
                </form>
                <input type="hidden" id="proNoticeId">
            </div>

        </div>
    </div>






    <div class="evaluationdisplay">
        <h3 style="margin-left:22px;">评分标准</h3>
       <hr>
        <form method="get" action="" id="updateEvalForm">
        <div><textarea style="font-size:16px;color:black;border-style:none;" readonly name="evaluationLevel" cols="130" rows="7" id="evaluationdetail"></textarea></div>
        <hr>
        <div id="updateEvaluationDiv" style="width:100%;height:200px;">
            <input type="button" value="修改或新建" id="updateEvaluationButton">
            <div style="width:100%;height:100px;display:none;" id="weightArea">
                <span>修改论坛，测验，考试占分比例</span>
                <hr>
                论坛:<select id="evaluationChatWeight" name="evaluationChatWeight">
                <option value="0.1">10%</option>
                <option value="0.15">15%</option>
                <option value="0.20" selected>20%</option>
                <option value="0.25">25%</option>
                <option value="0.3">30%</option>
                <option value="0.35">35%</option>
                <option value="0.4">40%</option>
                <option value="0.45">45%</option>
                <option value="0.5">50%</option>
                <option value="0.55">55%</option>
                <option value="0.6">60%</option>
                <option value="0.65">65%</option>
                <option value="0.7">70%</option>
            </select>
                &nbsp&nbsp
                测验:<select id="evaluationTestWeight" name="evaluationTestWeight">
                <option value="0.20">20%</option>
                <option value="0.25">25%</option>
                <option value="0.3" selected>30%</option>
                <option value="0.35">35%</option>
                <option value="0.4">40%</option>
                <option value="0.45">45%</option>
                <option value="0.5">50%</option>
                <option value="0.55">55%</option>
                <option value="0.6">60%</option>
                <option value="0.65">65%</option>
                <option value="0.7">70%</option>
            </select>
                &nbsp&nbsp
                考试:<select id="evaluationExamWeight" name="evaluationExamWeight">
                <option value="0.3">30%</option>
                <option value="0.35">35%</option>
                <option value="0.4">40%</option>
                <option value="0.45">45%</option>
                <option value="0.5" selected>50%</option>
                <option value="0.55">55%</option>
                <option value="0.6">60%</option>
                <option value="0.65">65%</option>
                <option value="0.7">70%</option>
            </select>
                <br>
                <input type="hidden" id="courseIdButton" name="courseId">
                <input type="button" value="提交" id="submitEvaluation">
            </div>
        </div>
        </form>

    </div>









    <div class="coursewaredisplay">
        <div class="coursewaretitle">课件<input type="button" value="回到目录总览" id="gobackutil" style="margin-left:30px;background:white;color:black;text-align:center;width:13%;border-style:none;display:none;">
        </div>
        <div class="coursewaredetail"><!--总的章节总览-->
        </div>
        <!--视频播放部分-->
        <div class="videoview">
            <input type="button" id="submitTextButton" value="上传本节文档" style="margin-left:39px;">&nbsp&nbsp<input type="button"  id="submitVideoButton" value="上传本节视频">
            <div id="submitTextDiv" style="margin-left:40px;width:50%;height:100px;border-style: outset;border-color: red;display:none;">
                <form id="submitTextForm" style="margin-top:5%;" action="upload" enctype="multipart/form-data" target="hidden_frame1" method="post">
                    <input type="hidden" value="text" name="sourceType">
                    <input type="hidden" value="-" name="poster">
                    <input type="hidden" name="belongUtilId" id="belongUtilId1">
                    <input type="file"  name="sourceSrc">
                    <input type="submit" id="submitTextButtonToServ" value="提交">
                </form>
                <iframe name="hidden_frame1" style="display:none;"></iframe>
            </div>
            <div id="submitVideoDiv" style="margin-left:40px;width:50%;height:200px;border-style: outset;border-color: red;display:none;">
               <form id="submitVideoForm" action="/upload" method="post" style="margin-top:5%;" enctype="multipart/form-data" target="hidden_frame" >
                   <input type="hidden" value="video" name="sourceType" id="sourceType">
                   选择视频封面:<input type="file"  name="posterSrc" id="posterSrc" ><br>
                   <input type="hidden" name="belongUtilId" id="belongUtilId">
                   上传视频:<input type="file"  name="videoSrc" id="videoSrc">
                <input type="submit" id="submitVideoButtonToServ" value="提交">
               </form>
                <iframe name="hidden_frame" style="display:none;"></iframe>
            </div>
            <hr>
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
        <input type="button" value="作业" style="font-size:21px;width:100px;height:45px;padding-top:6px;text-align:left;color:black;background:white;border-style:none;" id="homework">
        <input type="button" value="布置作业" style="font-size:21px;width:100px;height:45px;padding-top:6px;text-align:center;color:black;background:white;border-style:none;" id="prohomework">
        <input type="button" value="待发布作业" style="font-size:21px;width:120px;height:45px;padding-top:6px;text-align:center;color:black;background:white;border-style:none;" id="waitProHomeWork">
        <div class="testrough"><!--已发布作业总览-->
        </div>
        <div class="testWaitRough"><!--待发布作业总览-->
        </div>
        <div class="testEdit" style="display:none;margin-top:50px;width:100%;height:auto;"><!--试卷编辑区-->
            <div style="width:100%;height:auto;" id="testpaperblock"><!--试卷设计区-->
                <span style="display:block;color:black;font-size:19px;">试题内容:</span><hr><!--试卷头部提示-->
                <div class="alltestpaper"><!--整张试卷内容设计(包含所有题目部分和添加大题按钮)-->
                    <form class="writeTestDetail" id="testDetailForm"><!--仅包含题目部分 不包含添加大题按钮-->
                    </form>
                    <div class="addOneTestType"><!--添加大题-->
                        <span style="font-size:17px;color:black;float:left;margin-left:59%;margin-top:10px;">选择题型:&nbsp</span>
                        <select style="float:left;margin-top:10px;" id="ProbremType">
                            <option value="choose" selected>选择题</option>
                            <option value="fillin">填空题</option>
                            <option value="judge">判断题</option>
                        </select>
                        <input type="button" value="添加大题" style="background:green;color:white;text-align:center;padding-top:7px;width:110px;height:36px;border-style:none;border-radius:3px;float:left;margin-left:3%;" id="addBigTestType">
                    </div>
                </div>
        </div>
            <br>
            <hr>
            <div>
            <input type="button" value="提交" id="submitTestDetailButton">
            </div>
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
                <input type="hidden" name="studentId" id="teacherIds" value="<%=teacherId%>" />
                <input type="hidden" name="courseId" id="courseIds" />
                <input type="hidden" name="testId" id="testIds" />
                <div id="testPaper" style="width:100%;height:auto;margin-bottom:20px;background-color:rgb(251, 247, 247);">
                </div>
        </div>
        <hr style="width:100%">
        <div class="prohomeworkDiv" style="display:none;">
            <form action="" method="get" id="testBaseInfoForm"><!--这个表单仅包含试卷基本信息，没有具体内容-->
                <span style="color:black;font-size:18px;margin-left:2px;">选择单元</span>&nbsp&nbsp
                <select id="testbelongutil" name="belongChapterId" style="width:220px;">
                </select><br><hr>
                <input type="hidden" name="testType" id="testType" value="单元测验">
                输入作业标题：&nbsp<input type="text" style="width:190px;height:auto;color:grey;" id="testName" name="testName">
                <hr>
                输入起始时间：&nbsp<input type="Date" style="width:190px;height:auto;color:grey;" id="testStartTime" name="testStartTime"><hr>
                输入截止时间：&nbsp<input type="Date" style="width:190px;height:auto;color:grey;" id="testEndTime" name="testEndTime"><hr>
                测试/作业总分：<input type="number" placeholder="100" style="width:190px;height:auto;color:grey;" id="testAllMark" name="testAllMark"><hr>
                有效提交次数：&nbsp<input type="number" placeholder="3" style="width:190px;height:auto;color:grey;" id="inputSubmitCount" name="submitCount"><hr>
                限时(分钟)：&nbsp&nbsp&nbsp&nbsp&nbsp<input type="number" placeholder="20" style="width:192px;height:auto;color:grey;" id="limitTime" name="limitTime"><hr>
                作业/测试说明:
                <textarea cols="105" rows="5" id="testIntro" name="testIntro"></textarea>
                <input type="hidden" name="publicTeacherId" id="publicTeacherId">
            </form>
            <input type="button" value="提交" id="submitTestInfoButton" style="background:green;color:white;text-align:center;padding-top:7px;width:110px;height:36px;border-style:none;border-radius:3px;">
                <hr>
        </div>
    </div>




    <div class="examdisplay">
        <input type="button" value="布置期末测试" id="setExam" style="display:none;">
        <div id="checkInfo" style="display:none;">
        <div id="checkInfoDetail" >
            <div class="oneCheckInfo">
            </div>
        </div>
            <input type="button" value="查看试卷信息" id="lookExam">
            <input type="button" value="删除试卷" id="deleteExam">
        </div>
        <form method="post"  target="hidden_frame" action="submitExamInfo" id="examDesignArea" style="display:none;">
        <div id="examBaseInfo" style="width:80%;margin-left:10%;height:auto;">
            <div id="examTitle"><h3>布置期末测试</h3></div>
            考试说明:
            <div>
                <textarea id="examSuplement" name="examSuplement"  cols="80" rows="3" >
                </textarea>
            </div>
            <div >限制时间(分钟)：<input type="number" min="30" max="90" name="examLimitTime" id="examLimitTime" placeholder="60" /></div>
            <div >开始时间：<input type="date"  name="startTime" id="examStartTime" /></div>
            <div >截止时间：<input type="date"  name="endTime" id="examEndTime" /></div>
        </div>
        <div id="examDetail" style="">
                <input type="hidden" name="courseId" id="examPramOfCourseId">
                <div id="oneExamProblem" style="margin-top:4%;">
                    <div class="problemTitle" style="background-color:whitesmoke">
                        第一大题题目内容：
                       <div>
                           <textarea id="oneProblemTitle" cols="120" rows="4" name="examOneTitle">

                        </textarea>
                       </div>
                    </div>
                    第一大题答案详情:
                    <div class="problemAnswerArea" style="width:inherit;height:60%;background-color:papayawhip;" >
                        <textarea  style="width:inherit;height:70%;margin-top:5%;" id="problemAnswerSrcLookThroughOne" name="ExamOneAnswer" cols="120" rows="6"></textarea>
                    </div>
                </div>
            <div id="twoExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第二大题题目内容：
                   <div>
                       <textarea id="twoProblemTitle" cols="120" rows="4" name="examTwoTitle">

                        </textarea>
                   </div>
                </div>
                第二大题答案详情:
                <div class="problemAnswerArea" style="width:inherit;height:60%;background-color:papayawhip;" >
                    <textarea  style="width:inherit;height:70%;margin-top:5%;" id="problemAnswerSrcLookThroughTwo" name="examTwoAnswer" cols="120" rows="6"></textarea>
                </div>
            </div>
            <div id="threeExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第三大题题目内容：
                    <div>
                       <textarea id="threeProblemTitle" cols="120" rows="4" name="examThreeTitle">

                        </textarea>
                    </div>
                </div>
                第三大题答案详情:
                <div class="problemAnswerArea" style="width:inherit;height:60%;background-color:papayawhip;" >
                    <textarea  style="width:inherit;height:70%;margin-top:5%;" id="problemAnswerSrcLookThroughThree" name="examThreeAnswer" cols="120" rows="6"></textarea>
                </div>
            </div>
            <div id="fourExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第四大题题目内容：
                    <div>
                       <textarea id="fourProblemTitle" cols="120" rows="4" name="examFourTitle">

                        </textarea>
                    </div>
                </div>
                第四大题答案详情:
                <div class="problemAnswerArea" style="width:inherit;height:60%;background-color:papayawhip;" >
                    <textarea  style="width:inherit;height:70%;margin-top:5%;" id="problemAnswerSrcLookThroughFour" name="examFourAnswer" cols="120" rows="6"></textarea>
                </div>
            </div>
            <div id="fiveExamProblem" style="margin-top:4%;">
                <div class="problemTitle" style="background-color:whitesmoke">
                    第五大题题目内容：
                    <div>
                       <textarea id="fiveProblemTitle" cols="120" rows="4" name="examFiveTitle">

                        </textarea>
                    </div>
                </div>
                第五大题答案详情:
                <div class="problemAnswerArea" style="width:inherit;height:60%;background-color:papayawhip;" >
                    <textarea  style="width:inherit;height:70%;margin-top:5%;" id="problemAnswerSrcLookThroughFive" name="examFiveAnswer" cols="120" rows="6"></textarea>
                </div>
            </div>
            <input type="hidden" name="examPublishTeacherId" value="<%=teacherId%>">

                <iframe name="hidden_frame" style="display:none"></iframe>
                <div><input type="submit" value="提交" id="submitExamTeacherBtn" onclick="return submitForm();"></div>
        </div>
        </form>

















        <div id="studentExamInfo" style="display:none;">
            <form method="post" enctype="multipart/form-data" target="hidden_frame2" action="submitStudentExamScore">
                <input type="hidden" name="courseId" id="examOfCourseId">
                <input type="hidden" name="studentId" id="examOfStudentId">
                <div id="oneExamProblems" style="margin-top:4%;">
                    <div class="problemTitle" style="background-color:whitesmoke">
                        第一大题：
                        <p id="oneProblemTitles" style="word-break: break-all">

                        </p>
                        答案提示：
                        <div>
                            <textarea cols="120" rows="6" id="oneAnswerTip" readonly></textarea>
                        </div>
                    </div>
                    <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                        <div  style="width:inherit;height:95%;" id="AnswerSrcLookThroughOne">
                            <img style="width:60%;margin-left:20%;height:inherit;" id="oneImg">
                        </div>
                        <div>
                           评分: <input type="number" min="0" max="20" placeholder="10"  name="examScore" />
                        </div>
                    </div>
                </div>
                <div id="twoExamProblems" style="margin-top:4%;">
                    <div class="problemTitle" style="background-color:whitesmoke">
                        第二大题：
                        <p id="twoProblemTitles" style="word-break: break-all">

                        </p>
                        答案提示：
                        <div>
                            <textarea cols="120" rows="6" id="twoAnswerTip" readonly></textarea>
                        </div>
                    </div>
                    <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                        <div  style="width:inherit;height:95%;" id="AnswerSrcLookThroughTwo">
                            <img style="width:60%;margin-left:20%;height:inherit;" id="twoImg">
                        </div>
                        <div style="height:auto;">评分: <input type="number" min="0" max="20" placeholder="10"  name="examScore" /></div>
                    </div>
                </div>
                <div id="threeExamProblems" style="margin-top:4%;">
                    <div class="problemTitle" style="background-color:whitesmoke">
                        第三大题：
                        <p id="threeProblemTitles" style="word-break: break-all">

                        </p>
                        答案提示：
                        <div>
                            <textarea cols="120" rows="6" id="threeAnswerTip" readonly></textarea>
                        </div>
                    </div>
                    <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                        <div  style="width:inherit;height:95%;" id="AnswerSrcLookThroughThree">
                            <img style="width:60%;margin-left:20%;height:inherit;" id="threeImg">
                        </div>
                        <div style="height:auto;">评分: <input type="number" min="0" max="20" placeholder="10"  name="examScore" /></div>
                    </div>
                </div>
                <div id="fourExamProblems" style="margin-top:4%;">
                    <div class="problemTitle" style="background-color:whitesmoke">
                        第四大题：
                        <p id="fourProblemTitles" style="word-break: break-all">

                        </p>
                        答案提示：
                        <div>
                            <textarea cols="120" rows="6" id="fourAnswerTip" readonly></textarea>
                        </div>
                    </div>
                    <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                        <div  style="width:inherit;height:95%;" id="AnswerSrcLookThroughFour">
                            <img style="width:60%;margin-left:20%;height:inherit;" id="fourImg">
                        </div>
                        <div style="height:auto;">评分: <input type="number" min="0" max="20" placeholder="10"  name="examScore" /></div>
                    </div>
                </div>
                <div id="fiveExamProblems" style="margin-top:4%;">
                    <div class="problemTitle" style="background-color:whitesmoke">
                        第五大题：
                        <p id="fiveProblemTitles" style="word-break: break-all">

                        </p>
                        答案提示：
                        <div>
                            <textarea cols="120" rows="6" id="fiveAnswerTip" readonly></textarea>
                        </div>
                    </div>
                    <div class="problemAnswerArea" style="width:inherit;height:80%;background-color:papayawhip;" >
                        <div  style="width:inherit;height:95%;" id="AnswerSrcLookThroughFive">
                            <img style="width:60%;margin-left:20%;height:inherit;" id="fiveImg">
                        </div>
                        <div style="height:auto;">评分: <input type="number" min="0" max="20" placeholder="10"  name="examScore" /></div>
                    </div>
                </div>
                <iframe name="hidden_frame2" style="display:none"></iframe>
                <div><input type="submit" value="提交评分" id="submitScoreBtn" onclick="return submitExamScoreForm();"></div>
            </form>
        </div>
























        <div id="checkDivs" style="display:none;">
            审核结果：<select id="checkResultSelect">
            <option value="0">不通过</option>
            <option value="1">通过</option>
        </select><br>
            审核意见:<textarea id="checkResultRecommand" cols="120" rows="3"></textarea>
            <input type="button" value="确定" id="checkRight">
        </div>
        <div id="studentExamList" style="display:block;">
            <h4>学生提交列表:</h4>
           <table class="table" id="studentExamTable">
               <thead>
               <tr>
                   <th>学生姓名</th>
                   <th>学生账号</th>
                   <th>试卷类型</th>
               </tr>
               </thead>
               <tbody id="examBody">
               <!--<tr >
                   <td class="studentExamName" id="">张三</td>
                   <td class="studentExamAccount" id="">15051527</td>
                   <td class="studentExamType">期末测试</td>
               </tr>
               -->
               </tbody>
           </table>

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
</body>
</html>
<script>
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
    var arrys = [];//记录每道大题中小题的数目
    var  arryTypes = [];//记录每道大题的类型
    var trueAnswer = [];//记录每一道题目的正确答案
    var score = [];//记录每道大题中每小题的分值
    var courseids = "";
    var topicArry=[];
    var teststr = "";
    var chapterArr = [];
    var chapterArr1 = [];
    var teacherPic = "./img/" + "<%=teacherPic%>";
    var teacherId=<%=teacherId%>;

    $(document).ready(function () {
        $("#afterLogin").attr("title", "<%=teacherName%>");
        $("#Pic").attr("src",teacherPic);
    })
</script>
<script src="./js/courseWare.js"></script>
<script src="./js/functionList.js"></script>
<script src="./js/evaluation.js"></script>
<script src="./js/notice.js"></script>
<script src="./js/test.js"></script>
<script src="./js/exam.js"></script>
<script src="./js/comAndTopic.js"></script>
<script src="./js/upload.js"></script>
