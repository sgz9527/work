package com.xqq.controller;

import com.xqq.pojo.*;
import com.xqq.service.CourseService;
import com.xqq.service.CourseServiceImpl;
import com.xqq.service.InfoService;
import com.xqq.staticmethod.Files;
import com.xqq.staticmethod.getDate;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.multipart.commons.CommonsMultipartResolver;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
*
*@author xqq
*@date 2019/5/3 0003
*@description
*
*/
@Controller("CourseAction")
public class CourseAction {
    @Autowired
    CourseService courseServ;
    @Autowired
    CommonsMultipartResolver multipartResolver;
    @Autowired
    InfoService mInfoServ;
    HashMap<String, Object> map = new HashMap<String, Object>();
    List<HashMap<String, Object>> mapList = new LinkedList<HashMap<String, Object>>();

    @RequestMapping("indexCourse")
    /**
    返回参数：
    "courseId"
    "courseImgSrc"
    "courseState";
    "courseName";
    "school"
    "teachers";
    "fans";
    "comments";
    "courseIntroduce";
     */
    @ResponseBody
    public List<HashMap<String, Object>> showIndexCourse() {//展示首页课程
        System.out.println("response to show Index course");
        //定期处理消息记录，减少数据库存储
        courseServ.deleteMessage();
        mapList = courseServ.getIndexCourse();
        return mapList;
    }

    @RequestMapping("searchCourse")
    @ResponseBody
    public List<HashMap<String, Object>> searchCourse(String courseName) {
        System.out.println("response to show searchCourse");
        mapList = courseServ.getSearchCourse(courseName);
        return mapList;
    }

    @RequestMapping("getAllCourse")
    @ResponseBody
    public List<HashMap<String, Object>> getAllCourse(String type) {
        String ALL="all";
        System.out.println("response to getAllCourse");
        if (ALL.equals(type)) {
            mapList = courseServ.getAllCourse();
        } else {
            mapList = courseServ.getCourseByType(type);
        }
        return mapList;
    }
    @RequestMapping("getAllCourseCheck")
    @ResponseBody
    public JSONArray getAllCourseCheck() {
        System.out.println("response to getAllCourseCheck");
        return courseServ.getAllCourseCheck();
    }

    @RequestMapping("getTopTenCourse")
    @ResponseBody
    public List<HashMap<String, Object>> getTopTenCourse() {
        System.out.println("response to getTopTenCourse");
        mapList = courseServ.getTopTenCourse();
        return mapList;
    }

    @RequestMapping("getCourse")
    @ResponseBody
    public HashMap<String, Object> getUniqueCourse(HttpServletRequest request,Integer courseid) {
        HttpSession session=request.getSession();
        System.out.println("response to getCourse");
        //1472
        map = courseServ.getUniqueCourse(courseid,session);
        return map;
    }
    @RequestMapping("getExam")
    @ResponseBody
    public JSONObject getCourseAndExam(HttpServletRequest request, Integer courseid) {
        HttpSession session=request.getSession();
        System.out.println("response to getExam");
        //1091
        return courseServ.getCourseAndExam(session,courseid);
    }
    @RequestMapping("getStudentExamAnswer")
    @ResponseBody
    public JSONObject getStudentExamAnswer(Integer studentId, Integer courseId) {
        System.out.println("response to getStudentExamAnswer");
        //1223
        return courseServ.getCourseAndExam(studentId,courseId);
    }
    @RequestMapping("submitStudentExamScore")
    @ResponseBody
    public Object submitStudentExamScore(Integer []examScore,Integer studentId,Integer courseId) {
        //1239
        String result="";
        System.out.println("response to submitStudentExamScore,studentId = [" + studentId + "], courseId = [" + courseId + "]");
        result+=courseServ.submitStudentExamScore(examScore,studentId,courseId);
        return "<script>alert('"+result+"');</script>";
    }
    @RequestMapping("checkExam")
    @ResponseBody
    public String checkExam(Teach_courseKey tc) {
        //1147
        System.out.println("response to checkExam");
        return courseServ.checkExam(tc);
    }
    @RequestMapping("deleteExam")
    @ResponseBody
    public String deleteExam(Integer courseId) {
        System.out.println("response to deleteExam");
        //1147
        return courseServ.deleteExam(courseId);
    }
    @RequestMapping("submitExamInfo")
    @ResponseBody
    public Object submitExamInfo(HttpServletRequest request,Course course, String endTime, String startTime) {
        HttpSession session=request.getSession();
        System.out.println("response to submitExamInfo, endTime = [" + endTime + "], startTime = [" + startTime + "]");
        Date examEndTime= null;
        Date examStartTime=null;
        try {
            examEndTime = new SimpleDateFormat("yyyy-MM-dd").parse(endTime);
            examStartTime = new SimpleDateFormat("yyyy-MM-dd").parse(startTime);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        course.setExamStartTime(examStartTime);
        course.setExamEndTime(examEndTime);
        String errorMsg="";
        //1147
        if(courseServ.submitExamInfo(course,session)==1){
            errorMsg="success";
        }else{
            errorMsg="error";
        }
        return "<script>alert('"+errorMsg+"')</script>";
    }
    @RequestMapping(value = "submitExam", method = RequestMethod.POST)
    @ResponseBody
    public Object submitExam(HttpServletRequest request) {
        System.out.println("response to submitExam");
        String result="";
        HttpSession session=request.getSession();
        Student stu=(Student)session.getAttribute("studentInfo");
        if(stu==null){
            result="login error:no user is login";
            return "<script>alert('"+result+"');</script>";
        }else{
            Integer courseId=-1;
            String one="",two="",three="",four="",five="";
            //第一步先把答案图片存入文件夹
            //第二步路径存入数据库,submitExam位于1119行
            if (multipartResolver.isMultipart(request)){
                MultipartHttpServletRequest multiRequest = (MultipartHttpServletRequest) request;
                courseId=Integer.parseInt(multiRequest.getParameter("courseId"));
                Iterator<String> iter = multiRequest.getFileNames();
                String imgPath = request.getSession().getServletContext().getRealPath("/img");
                 int count=0;
                while (iter.hasNext()) {
                    count++;
                    //取得上传文件
                    MultipartFile file = multiRequest.getFile(iter.next());
                    if (file != null) {
                        String path = "";
                        //取得当前上传文件的文件名称
                        String myFileName = file.getOriginalFilename();
                        //如果名称不为“”,说明该文件存在，否则说明该文件不存在
                        if (myFileName.trim() != "") {
                            //重命名上传后的文件名
                            String fileName = "Upload" + file.getOriginalFilename();
                                //先检测fileName是否在指定文件夹下存在
                                String fileNameNew = Files.isExistName(imgPath, fileName);
                            switch(count){
                                case 1:one=fileNameNew;break;
                                case 2:two=fileNameNew;break;
                                case 3:three=fileNameNew;break;
                                case 4:four=fileNameNew;break;
                                case 5:five=fileNameNew;break;
                                default:break;
                            }
                                path = imgPath + "\\" + fileNameNew;
                            File localFile = new File(path);
                            try {
                                file.transferTo(localFile);
                            } catch (IOException e) {
                                e.printStackTrace();
                                result += "\\\\\\\\\\" + e.getMessage();
                            }
                        }
                    }
                }
            }
            result+=courseServ.submitExam(stu.getStudentId(),courseId,one,two,three,four,five);
        }
        return "<script>alert('"+result+"');</script>";
    }

    @RequestMapping("getTeacher")
    @ResponseBody
    public List<HashMap<String, Object>> getTeacher(Integer courseid) {
        System.out.println("response to getTeacher");
        mapList = courseServ.getTeacher(courseid);
        return mapList;
    }

    @RequestMapping("chooseCourse")
    @ResponseBody
    public HashMap<String, Object> chooseCourse(HttpServletRequest request,Integer courseid) {
        HttpSession session=request.getSession();
        System.out.println("response to chooseCourse");
        if (courseServ.chooseCourse(courseid,session)) {
            map.put("result", "ok");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @RequestMapping("studentCourse")
    @ResponseBody
    public List<HashMap<String, Object>> studentCourse(HttpServletRequest request,Integer studentId) {
        HttpSession session=request.getSession();
        Student student = (Student) session.getAttribute("studentInfo");
        System.out.println("response to studentCourse>>>>>>");
        if (student != null) {
            System.out.println("CourseAction.studentCourse:studentId:"+student.getStudentId());
            mapList = courseServ.getStudentCourse(student.getStudentId());
        }
        else {
            System.out.println("CourseAction.studentCourse:student is null,use studentId to select!");
            Student stu1=mInfoServ.getStudent(studentId);
            mapList = courseServ.getStudentCourse(studentId);
            session.removeAttribute("studentInfo");
            session.setAttribute("studentInfo",stu1);
            Student stu2=(Student)session.getAttribute("studentInfo");
            System.out.println("CourseAction.studentCourse,sessionStu.name:"+stu2.getStudentName());
        }
        return mapList;
    }

    @RequestMapping(value = "addChapterAndUtil", method = RequestMethod.POST)
    @ResponseBody
    public Object addChapterAndUtil(
            Integer chapterNum, Integer[] belongCourseId, String[] chapterTitle,
            String[] chapterName, Integer[] chapterOrder, Integer[] belongChapterId,
            String[] utilTitle, String[] utilName, Integer[] utilOrder
    ) {
        String errorMsg = "";
        System.out.println("response to addChapterAndUtil");
        //（章+节）集合
        ArrayList<ChapterAndUtilList> cauList = new ArrayList<>();
        for (int i = 0; i < chapterNum; i++) {
            Chapter chapter = new Chapter();
            //此时的belongCourseId仅仅是checkId
            chapter.setBelongCourseId(belongCourseId[i]);
            chapter.setChapterName(chapterName[i]);
            chapter.setChapterOrder(chapterOrder[i]);
            chapter.setChapterTitle(chapterTitle[i]);
            ArrayList<Util> utilList = new ArrayList<Util>();
            for (int j = 0; j < utilTitle.length; j++) {
                if (i == belongChapterId[j]) {
                    Util util = new Util();
                    //此时的belongChapterId仅仅是下标
                    util.setBelongChapterId(belongChapterId[j]);
                    util.setUtilName(utilName[j]);
                    util.setUtilOrder(utilOrder[j]);
                    util.setUtilTitle(utilTitle[j]);
                    utilList.add(util);
                }
            }
            ChapterAndUtilList cau = new ChapterAndUtilList();
            cau.setChapter(chapter);
            cau.setUtilList(utilList);
            cauList.add(cau);
        }
        //位于第625行
        errorMsg = courseServ.addChapterAndUtil(cauList);
        if (errorMsg.equals("")) {
            errorMsg = "submit success";
        }
        return "<script>alert('" + errorMsg + "');</script>";
    }

    @RequestMapping("teacherCourse")
    @ResponseBody
    public List<HashMap<String, Object>> teacherCourse(HttpServletRequest request) {
        HttpSession session=request.getSession();
        Teacher teacher = (Teacher) session.getAttribute("teacherInfo");
        System.out.println("response to teacherCourse");
        if (teacher != null) {
            System.out.println("CourseAction.teacherCourse:teacher.teacherName:"+teacher.getTeacherName());
            mapList = courseServ.getTeacherCourse(teacher.getTeacherId());
        }
        else{
            System.out.println("CourseAction.teacherCourse:teacher is null!");
        }
        return mapList;
    }

    @RequestMapping("applicateCourseState")
    @ResponseBody
    public List<HashMap<String, Object>> getApplicateCourseState(Integer teacherId) {
        System.out.println("response to applicateCourseState");
        mapList = courseServ.getTeacherCheckCourse(teacherId);
        return mapList;
    }
    @RequestMapping("setCheckCourseState")
    @ResponseBody
    public HashMap<String, Object> setCheckCourseState(CheckCourse checkCourse) {
        System.out.println("CourseAction.setCheckCourseState");
        map = courseServ.setCheckCourseState(checkCourse);
        return map;
    }

    @RequestMapping(value = "applicateCourse", method = RequestMethod.POST)
    @ResponseBody
    public Object applicateCourse(HttpServletRequest request) {
        String errorMsg = "";
        if (multipartResolver.isMultipart(request)) {
            //转换成多部分request
            MultipartHttpServletRequest multiRequest = (MultipartHttpServletRequest) request;
            String courseName = multiRequest.getParameter("courseName");
            Integer teacherId = Integer.parseInt(multiRequest.getParameter("teacherId"));
            Integer belongSchId = Integer.parseInt(multiRequest.getParameter("belongSchId"));
            String courseIntroduce = multiRequest.getParameter("courseIntroduce");
            String courseType = multiRequest.getParameter("courseType");
            System.out.println("curseType:" + courseType);
            String teacherPhone = multiRequest.getParameter("teacherPhone");
            String posterSrc = "";
            String introduceSrc = "";
            //生成一个CheckCourse实体类对象
            CheckCourse cc = new CheckCourse();
            cc.setBelongSchId(belongSchId);
            cc.setCourseIntroduce(courseIntroduce);
            cc.setCourseName(courseName);
            cc.setCourseType(courseType);
            cc.setTeacherPhone(teacherPhone);
            cc.setTeacherId(teacherId);
            //取得request中的所有文件名
            Iterator<String> iter = multiRequest.getFileNames();
            //统计次数，第一次为视频封面图片，第二次为视频src
            int count = 0;
            ArrayList<MultipartFile> MultipartFileList = new ArrayList<MultipartFile>();
            ArrayList<String> FilePathList = new ArrayList<String>();
            String videoPath = request.getSession().getServletContext().getRealPath("/video");
            String imgPath = request.getSession().getServletContext().getRealPath("/img");
            while (iter.hasNext()) {
                count++;
                //取得上传文件
                MultipartFile file = multiRequest.getFile(iter.next());
                if (file != null) {
                    String path = "";
                    MultipartFileList.add(file);
                    //取得当前上传文件的文件名称
                    String myFileName = file.getOriginalFilename();
                    //如果名称不为“”,说明该文件存在，否则说明该文件不存在
                    if (myFileName.trim() != "") {
                        //重命名上传后的文件名
                        String fileName = "Upload" + file.getOriginalFilename();
                        if (count == 1) {
                            //先检测fileName是否在指定文件夹下存在
                            String fileNameNew = Files.isExistName(imgPath, fileName);
                            path = imgPath + "\\" + fileNameNew;
                            cc.setPosterSrc(fileNameNew);
                        } else if (count == 2) {
                            String fileNameNew = Files.isExistName(videoPath, fileName);
                            path = videoPath + "\\" + fileNameNew;
                            cc.setIntroduceSrc(fileNameNew);
                        }
                        //将path加入集合
                        FilePathList.add(path);
                    }
                } else {
                    errorMsg += "\\\\\\\\emptyfile";
                }
            }
            if (count == 0) {
                errorMsg += "\\\\emptyfile";
            }
            boolean isAdd = false;
            if (cc != null) {
                isAdd = courseServ.addCheckCourse(cc);
            }
            if (isAdd) {
                for (int i = 0; i < FilePathList.size(); i++) {
                    File localFile = new File(FilePathList.get(i));
                    try {
                        MultipartFileList.get(i).transferTo(localFile);
                    } catch (IOException e) {
                        e.printStackTrace();
                        errorMsg += "\\\\\\\\\\" + e.getMessage();
                    }
                }
            } else {
                errorMsg += "\\\\\\\\errorAddApplicateCourse";
            }
        }
        if (errorMsg.equals("")) {
            errorMsg = "submit success";
        }
        return "<script>alert('" + errorMsg + "')</script>";
    }

    @RequestMapping("studentDeleteCourse")
    @ResponseBody
    public HashMap<String, Object> deleteCourse(HttpServletRequest request ,Integer deleteCourseId) {
        HttpSession session=request.getSession();
        Student student = (Student) session.getAttribute("studentInfo");
        if (student == null) {
            map.put("result", "system error,not login");
        } else {
            int res = courseServ.deleteCourse(deleteCourseId, student.getStudentId());
            if (res == 2) {
                map.put("result", "无法删除超过三天时间的课程!");
            } else if (res == 1) {
                map.put("result", "删除成功!");
            } else {
                map.put("result", "system error");
            }
        }
        return map;
    }

    @RequestMapping("getReCommandCourse")
    @ResponseBody
    public List<HashMap<String, Object>> reCommandCourse(HttpServletRequest request) {
        HttpSession session=request.getSession();
        Student student = (Student) session.getAttribute("studentInfo");
        System.out.println("response to getReCommandCourse");
        if (student != null) {
            System.out.println("CourseAction.reCommandCourse,student.name:"+student.getStudentName());
            mapList = courseServ.getReCommandCourse(student.getStudentId());
        }
        return mapList;
    }

    @RequestMapping("noticeinfo")
    @ResponseBody
    public List<HashMap<String, Object>> getNotices(Integer courseid) {
        mapList = courseServ.getNotices(courseid);
        return mapList;
    }

    @RequestMapping("evaluationinfo")
    @ResponseBody
    public HashMap<String, Object> getEvaluation(HttpServletRequest request,Integer courseid) {
        HttpSession session=request.getSession();
        map = courseServ.getUniqueCourse(courseid,session);
        return map;
    }

    @RequestMapping("chapterinfo")
    /**
      ["chapterId"]
      ["chapterId"]
      ["chapterId"]  ["titleNumberName"]["titleName"]
      ["chapterId"]
     */
    @ResponseBody
    public List<HashMap<String, Object>> getChapters(Integer courseid) {
        mapList = courseServ.getChapters(courseid);
        return mapList;
    }

    @RequestMapping("utilinfo")
    /**
     utilId   utilDetailTitle  utilDetailName+
                                   utilDetailVideoNumber
                                  utilDetailTextNumber
     */
    @ResponseBody
    public List<HashMap<String, Object>> getUtils(Integer chapterId) {
        mapList = courseServ.getUtils(chapterId);
        return mapList;
    }

    @RequestMapping(value = "upload", method = RequestMethod.POST)
    @ResponseBody
    public Object fileUpload(HttpServletRequest request) {
        String errorMsg = "";
        if (multipartResolver.isMultipart(request)) {
            //转换成多部分request
            MultipartHttpServletRequest multiRequest = (MultipartHttpServletRequest) request;
            String belongUtilIdStr = multiRequest.getParameter("belongUtilId");
            Integer belongUtilId = Integer.parseInt(belongUtilIdStr);
            String sourceType = multiRequest.getParameter("sourceType");
            String poster = "-";
            //生成一个source实体类对象
            Source source = new Source();
            source.setBelongUtilId(belongUtilId);
            source.setSourceType(sourceType);
            //取得request中的所有文件名
            Iterator<String> iter = multiRequest.getFileNames();
            //统计次数，第一次为视频封面图片，第二次为视频src
            int count = 0;
            ArrayList<MultipartFile> MultipartFileList = new ArrayList<MultipartFile>();
            ArrayList<String> FilePathList = new ArrayList<String>();
            String videoPath = request.getSession().getServletContext().getRealPath("/video");
            String textPath = request.getSession().getServletContext().getRealPath("/Texts");
            String imgPath = request.getSession().getServletContext().getRealPath("/img");
            while (iter.hasNext()) {
                count++;
                //取得上传文件
                MultipartFile file = multiRequest.getFile(iter.next());
                if (file != null) {
                    String path = "";
                    MultipartFileList.add(file);
                    //取得当前上传文件的文件名称
                    String myFileName = file.getOriginalFilename();
                    //如果名称不为“”,说明该文件存在，否则说明该文件不存在
                    if (myFileName.trim() != "") {
                        //重命名上传后的文件名
                        String fileName = "Upload" + file.getOriginalFilename();
                        if (count == 1) {
                            if (sourceType.equals("video")) {
                                //先检测fileName是否在指定文件夹下存在
                                String fileNameNew = Files.isExistName(imgPath, fileName);
                                path = imgPath + "\\" + fileNameNew;
                                source.setPoster(fileNameNew);
                            } else {
                                String fileNameNew = Files.isExistName(textPath, fileName);
                                path = textPath + "\\" + fileNameNew;
                                //如果不是视频而是文档，则默认没有图片
                                source.setPoster(poster);
                                source.setSourceSrc(fileNameNew);
                            }
                        } else if (count == 2) {
                            String fileNameNew = Files.isExistName(videoPath, fileName);
                            path = videoPath + "\\" + fileNameNew;
                            source.setSourceSrc(fileNameNew);
                        }
                        //将path加入集合
                        FilePathList.add(path);
                    }
                } else {
                    errorMsg += "\\\\\\\\emptyfile";
                }
            }
            if (count == 0) {
                errorMsg += "\\\\emptyfile";
            }
            boolean isAdd = false;
            System.out.println("source.belongUtilId:" + source.getBelongUtilId() + "  source.sourceType:" + source.getSourceType() + "  source.sourceSrc:" + source.getSourceSrc() + "  source.poster:" + source.getPoster());
            if (source != null) {
                isAdd = courseServ.addSource(source);
            }
            if (isAdd) {
                for (int i = 0; i < FilePathList.size(); i++) {
                    File localFile = new File(FilePathList.get(i));
                    try {
                        MultipartFileList.get(i).transferTo(localFile);
                    } catch (IOException e) {
                        e.printStackTrace();
                        errorMsg += "\\\\\\\\\\\\" + e.getMessage();
                    }
                }
            } else {
                errorMsg += "\\\\\\\\errorAddSource";
            }
        }
        if (errorMsg.equals("")) {
            errorMsg = "success";
        }
        return "<script>alert('" + errorMsg + "')</script>";
    }

    @RequestMapping(value = "uploadText", method = RequestMethod.POST)
    @ResponseBody
    public Object uploadText(@RequestParam("belongUtilId") Integer belongUtilId, @RequestParam("sourceType") String sourceType, @RequestParam("sourceSrc") MultipartFile sourceSrc) {
        String errorMsg = "";
        System.out.println("belongUtilId:" + belongUtilId + "||sourceType:" + sourceType);
        if (errorMsg.equals("")) {
            errorMsg = "success";
        }
        return "<script>alert('" + errorMsg + "')</script>";
    }

    @RequestMapping("getSource")
    @ResponseBody
    public List<HashMap<String, Object>> getSources(Integer utilId) {
        mapList = courseServ.getSources(utilId);
        return mapList;
    }

    @RequestMapping("getTest")
    @ResponseBody
    public List<HashMap<String, Object>> getTest(Integer courseId,HttpServletRequest request) {
        HttpSession session=request.getSession();
        mapList = courseServ.getTest(courseId,session);
        System.out.println("courseId:" + courseId + "getTest:" + mapList.size());
        return mapList;
    }
    @RequestMapping("getWaitTest")
    @ResponseBody
    public List<HashMap<String, Object>> getWaitTest(Integer courseId) {
        mapList = courseServ.getWaitTest(courseId);
        System.out.println("courseId:" + courseId + "getWaitTest:" + mapList.size());
        return mapList;
    }

    @RequestMapping("submitTest")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> submitTest(Test_score score) {
        System.out.println("submitTest--getStudentScore:" + score.getScore());
        map = courseServ.submitTest(score);
        return map;
    }

    @RequestMapping("submitTestBaseInfo")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> submitTestBase(Test test) {
        System.out.println("submitTestBaseInfo:" + test.getTestName());
        map = courseServ.submitTestBase(test);
        return map;
    }

    @RequestMapping("messageInfo")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> getMessageInfo(Integer studentId) {
        System.out.println("response to getMessageInfo,studentId:" + studentId);
        map = courseServ.getMessageInfo(studentId);
        return map;
    }
    @RequestMapping("setMessageState")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> setMessageState(HttpServletRequest request,@RequestParam("messageIdList") String messageIdList) {
        System.out.println("response to setMessageState,messageId.toString:" +messageIdList);
        String emptyStr="[]";
        if(!emptyStr.equals(messageIdList)){
            String newStr="";
            HttpSession session=request.getSession();
            Student stu=(Student)session.getAttribute("studentInfo");
            Integer studentId=-1;
            if(stu!=null){
                studentId=stu.getStudentId();
            }
            for(int i=0;i< messageIdList.length();i++) {
                if(messageIdList.charAt(i)!='['&&messageIdList.charAt(i)!=' '&&messageIdList.charAt(i)!=']'){
                    newStr+=messageIdList.charAt(i);
                }
            }
            ArrayList<String> messageId= CourseServiceImpl.segment(newStr,',');
            ArrayList<Integer> messageIdList1=new ArrayList<>();
            for(int i=0;i<messageId.size();i++){
                messageIdList1.add(Integer.parseInt(messageId.get(i).toString()));
            }
            //855行
            if(messageIdList1.size()!=0){
                if(courseServ.setMessageState(messageIdList1,studentId)){
                    map.put("result","ok");
                }
                else{
                    map.put("result","error");
                }
            }
        }
        return map;
    }

    @RequestMapping("submitTestDetailInfo")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> submitDetailTestInfo(//这里belongTestProblemId不是真的id，需要转换
                                                        Integer[] belongTestId, short[] testProblemType, Integer[] testProblemOrder,
                                                        String[] testProblemTitle, Integer[] belongTestProblemId, short[] smallProblemType,
                                                        short[] smallProblemScore, String[] smallProblemADetail, String[] smallProblemBDetail,
                                                        String[] smallProblemCDetail, String[] smallProblemDDetail, String[] smallProblemTip, String[] smallProblemTrueAnswer, String[] smallProblemTitle) {
        ArrayList<TestProblem> testProblemList = new ArrayList<TestProblem>();
        int i ;
        //生成大题实体类集合
        for (i = 0; i < testProblemOrder.length; i++)
        {
            TestProblem tesp = new TestProblem();
            tesp.setBelongTestId(belongTestId[i]);
            tesp.setTestProblemOrder(testProblemOrder[i]);
            tesp.setTestProblemTitle(testProblemTitle[i]);
            tesp.setTestProblemType(testProblemType[i]);
            testProblemList.add(tesp);
        }
        System.out.println("testProblemList.size:" + testProblemList.size());
        ArrayList<smallProblem> smallProblemList = new ArrayList<smallProblem>();
        //生成小题实体类集合
        for (i = 0; i < belongTestProblemId.length; i++) {
            smallProblem smp = new smallProblem();
            //此处只是为了识别是属于当前的哪一道大题，并非是最终数据库中的TestProblemId
            smp.setBelongTestProblemId(belongTestProblemId[i]);
            if (smallProblemADetail[i] != "null") {
                smp.setSmallProblemADetail(smallProblemADetail[i]);
                smp.setSmallProblemBDetail(smallProblemBDetail[i]);
                smp.setSmallProblemCDetail(smallProblemCDetail[i]);
                smp.setSmallProblemDDetail(smallProblemDDetail[i]);
            }
            smp.setSmallProblemScore(smallProblemScore[i]);
            if (smallProblemTip[i] != "null")
                smp.setSmallProblemTip(smallProblemTip[i]);
            smp.setSmallProblemTitle(smallProblemTitle[i]);
            smp.setSmallProblemTrueAnswer(smallProblemTrueAnswer[i]);
            smp.setSmallProblemType(smallProblemType[i]);
            smallProblemList.add(smp);
        }
        System.out.println("smallProblemList.size:" + smallProblemList.size());
        //接下来生成完整(大题+小题)集合
        ArrayList<testProblemAndSmall> bigAndSmallList = new ArrayList<testProblemAndSmall>();
        for (i = 0; i < testProblemList.size(); i++) {
            testProblemAndSmall tpas = new testProblemAndSmall();
            ArrayList<smallProblem> smallList = new ArrayList<smallProblem>();
            Integer order = testProblemList.get(i).getTestProblemOrder();
            for (int j = 0; j < smallProblemList.size(); j++) {
                if (smallProblemList.get(j).getBelongTestProblemId().equals(order)) {
                    smallList.add(smallProblemList.get(j));
                }
            }
            tpas.setSmallProblemList(smallList);
            tpas.setTestProblem(testProblemList.get(i));
            bigAndSmallList.add(tpas);
        }
        System.out.println("bigAndSmallList.size():" + bigAndSmallList.size());
        map = courseServ.submitDetailTest(bigAndSmallList);
        map.put("result", "ok");
        return map;
    }

    @RequestMapping("getScoreAndSubmitCount")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> getScoreAndSubmitCount(Integer studentId, Integer testId) {
        System.out.println("getScoreAndSubmitCount--getStudentId:" + studentId);
        map = courseServ.getScoreAndCount(studentId, testId);
        return map;
    }

    @RequestMapping("getScore")
    /**

     */
    @ResponseBody
    public List<HashMap<String, Object>> getScore(Integer studentId) {
        System.out.println("getScore--getStudentId:" + studentId);
        mapList = courseServ.getScore(studentId);
        return mapList;
    }

    @RequestMapping("scoreInfo")
    /**

     */
    @ResponseBody
    public List<HashMap<String, Object>> getScoreInfoByTeacherId(Integer teacherId) {
        System.out.println("getScoreInfoByTeacherId--getTeacherId:" + teacherId);
        mapList = courseServ.getScoreInfoByTeacherId(teacherId);
        return mapList;
    }

    @RequestMapping("updateNotice")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> updateNotice(Notice notice) {
        notice.setSystemTime(new Date());
        System.out.println("updateNotice--getnoticeDetail:" + notice.getTime());
        System.out.println("systemTime" + getDate.getHMS());
        map = courseServ.updateNotice(notice);
        return map;
    }

    @RequestMapping("addSaveNotice")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> addSaveNotice(Notice notice) {
        notice.setSystemTime(new Date());
        notice.setNoticeState("wait");
        System.out.println("addSaveNotice--getnoticeDetail:" + notice.getNoticeDetail());
        map = courseServ.addSaveNotice(notice);
        return map;
    }

    @RequestMapping("proNotice")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> proNotice(Notice notice) {
        notice.setNoticeState("finish");
        System.out.println("addSaveNotice--getnoticeId:" + notice.getNoticeId());
        map = courseServ.proNotice(notice);
        return map;
    }

    @RequestMapping("deleteNotice")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> deleteNotice(Integer noticeId) {
        System.out.println("deleteNotice--noticeId:" + noticeId);
        map = courseServ.deleteNotice(noticeId);
        return map;
    }

    @RequestMapping("updateEvaluation")
    /**

     */
    @ResponseBody
    public HashMap<String, Object> updateEvaluation(Course course) {
        System.out.println("updateEvaluation--courseId:" + course.getCourseId());
        map = courseServ.updateCourse(course);
        return map;
    }
    @RequestMapping("getFinalExamScore")
    /**

     */
    @ResponseBody
    public JSONArray getFinalExamSocre(Integer studentId) {
        System.out.println("response to CourseAction.getFinalExamSocre");
        return courseServ.getFinalExamScore(studentId);
    }


}
