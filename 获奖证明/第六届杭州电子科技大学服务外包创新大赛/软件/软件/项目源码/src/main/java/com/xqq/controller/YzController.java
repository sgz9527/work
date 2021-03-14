package com.xqq.controller;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xqq.constant.Constants;
import com.xqq.pojo.Manager;
import com.xqq.pojo.Student;
import com.xqq.pojo.Teacher;
import com.xqq.service.CourseService;
import com.xqq.service.InfoService;
import com.xqq.service.LoginService;
import com.xqq.staticmethod.Files;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.multipart.commons.CommonsMultipartResolver;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

/**
 * @author Administrator
 */
@Controller("loginAction")
public class YzController {
    @Autowired
    LoginService mLoginService;
    @Autowired
    InfoService mInfoService;
    @Autowired
    CommonsMultipartResolver mMultipartResolver;
    @Autowired
    CourseService mCourseServ;
    private HashMap<String, Object> mMap = new HashMap<>();

    /**************学生业务**************/
    @RequestMapping("studentLogin")
    public String studentLogin(HttpServletRequest request,Student stu) {
        System.out.println("studentAccount:" + stu.getStudentAccount() + "  studentPassword:" + stu.getStudentPassword());
        Student stu1 = mLoginService.LoginStudent(stu.getStudentAccount(), stu.getStudentPassword());
        if (stu1 != null) {
            System.out.println("Login Pass!");
            System.out.println("studentName:" + stu1.getStudentName());
            HttpSession mysession=request.getSession();
            mysession.setAttribute("studentInfo", (Student) stu1);
            //获取登录时间，检测该用户所有作业以及考试是否距离截止时间仅剩2天
            mCourseServ.detectAllHomeWorkEndTime(stu1.getStudentId());
            return "index";
        } else {
            System.out.println("forbidden Login!");
            return "fail";
            //map.put("LoginStudentInfo","account or password wrong,check and try again!");
        }
    }
    @RequestMapping("studentRegiste")
    @ResponseBody
    public String studentRegist(Student stu) {
        if (stu != null) {
            if(mLoginService.studentRegist(stu)==1){
                return "<script>alert('success');</script>";
            }else{
                return "<script>alert('fail');</script>";
            }
        } else {
            return "<script>alert('success');</script>";
        }
    }
    /**************学生业务**************/


    /**************教师登录业务**************/
    @RequestMapping("teacherLogin")
    public String teacherLogin(HttpServletRequest request,Teacher teach) {
        Teacher teacher1 = mLoginService.LoginTeacher(teach.getTeacherAccount(), teach.getTeacherPassword());
        if (teacher1 != null) {
            System.out.println("Login Pass!");
            System.out.println("teacherName:" + teacher1.getTeacherName());
            HttpSession mysession=request.getSession();
            mysession.setAttribute("teacherInfo", (Teacher) teacher1);
            Teacher tea = (Teacher) mysession.getAttribute("teacherInfo");
            System.out.println("sessionTeach:" + tea.getTeacherId());
            return "index";
        } else {
            System.out.println("forbidden Login!");
            return "fail";
        }
        //return map;
    }

    /**************教师登录业务**************/
    /**************管理员登录业务**************/
    @RequestMapping("adminLogin")
    public String adminLogin(HttpServletRequest request,Manager admin) {
        HttpSession session=request.getSession();
        Manager mana1=(Manager)session.getAttribute("adminInfo");
        if(mana1!=null){
            return "fail";
        }
        Manager manager = mLoginService.loginAdmin(admin.getAccount(), admin.getPassword());
        if (manager != null) {
            System.out.println("admin Login Pass!");
            System.out.println("adminName:" + manager.getManagerName());
            HttpSession mysession=request.getSession();
            mysession.setAttribute("adminInfo", (Manager) manager);
            return "manager";
        } else {
            System.out.println("forbidden admin Login!");
            return "fail";
        }
    }


    @RequestMapping("faceLogin")
    @ResponseBody
    public String adminFaceLogin(HttpServletRequest request,String baseData) throws Exception {
        HttpSession session=request.getSession();
        Manager mana1=(Manager)session.getAttribute("adminInfo");
        if(mana1!=null){
            return "alreadyLogin!";
        }else {
            Manager manager = mLoginService.faceAdminLoginBaidu(baseData);
            if (manager != null) {
                //写入session保存
                System.out.println("adminLoginByFace pass,adminName:" + manager.getManagerName());
                HttpSession mysession = request.getSession();
                mysession.setAttribute("adminInfo", (Manager) manager);
                return "manager";
            } else {
                System.out.println("forbidden admin Login by face!");
                return "fail";
            }
        }
    }
    @RequestMapping("faceAdd")
    @ResponseBody
    public String FaceAdd(HttpServletRequest request,String baseData) throws Exception {
        HttpSession session=request.getSession();
        Student stu=(Student)session.getAttribute("studentInfo");
        String result="-1";
        if(stu!=null){
            result=mLoginService.faceAdd(baseData, Constants.STUDENT_GROUP_ID,stu.getStudentAccount());
        }else{
            Teacher teacher=(Teacher)session.getAttribute("teacherInfo");
            if(teacher!=null){
                result=mLoginService.faceAdd(baseData, Constants.TEACHER_GROUP_ID,teacher.getTeacherAccount());
            } else{
                Manager manager=(Manager)session.getAttribute("adminInfo");
                if(manager!=null){
                    result=mLoginService.faceAdd(baseData, Constants.ADMIN_GROUP_ID,manager.getAccount());
                }else{
                    System.out.println("YzController.FaceAdd,error:there is no user has already login");
                    return "no user has already login,can not add face";
                }
            }
        }
        if(!result.equals("-1")){
            JsonObject jsonObj = (new JsonParser().parse(result)).getAsJsonObject();
            result=jsonObj.get("error_msg").getAsString();
        }
        return result;
    }
    /**************管理员登录业务**************/
    @RequestMapping("teacherInfo")
    @ResponseBody
    public HashMap<String, Object> teacherInfo(Integer teacherId) {

        return mInfoService.getTeacherInfo(teacherId);
    }

    /**************账户退出登录**************/
    @RequestMapping("exitSystem")
    @ResponseBody
    public HashMap<String, Object> exitSystem(HttpServletRequest request) {
        HttpSession session=request.getSession();
        if (session.getAttribute("studentInfo") != null) {
            session.removeAttribute("studentInfo");
            System.out.println("退出成功!");
        } else if (session.getAttribute("teacherInfo") != null) {
            session.removeAttribute("teacherInfo");
            System.out.println("退出成功!");
        }
        else if(session.getAttribute("adminInfo") != null){
            session.removeAttribute("adminInfo");
            System.out.println("退出成功!");
        }
        mMap.put("exit", "ok");
        return mMap;
    }

    @RequestMapping(value = "updateStudentInfo", method = RequestMethod.POST)
    @ResponseBody
    public Object updateStuInfo(HttpServletRequest request) {
        String msg = "";
        if (mMultipartResolver.isMultipart(request)) {
            //转换成多部分request
            MultipartHttpServletRequest multiRequest = (MultipartHttpServletRequest) request;
            String studentName = multiRequest.getParameter("studentName");
            String studentPassword = multiRequest.getParameter("studentPassword");
            String studentSex = multiRequest.getParameter("studentSex");
            String studentEmail = multiRequest.getParameter("studentEmail");
            Integer studentId = Integer.parseInt(multiRequest.getParameter("studentId"));
            String studentIntroduce = multiRequest.getParameter("studentIntroduce");
            Student stu = new Student();
            stu.setStudentEmail(studentEmail);
            stu.setStudentPassword(studentPassword);
            stu.setStudentIntroduce(studentIntroduce);
            stu.setStudentSex(studentSex);
            stu.setStudentName(studentName);
            stu.setStudentId(studentId);
            Iterator<String> iter = multiRequest.getFileNames();
            String imgPath = request.getSession().getServletContext().getRealPath("/img");
            System.out.println("YzController.updateStuInfo:imgPath:"+imgPath);
            String path = "";
            MultipartFile file = null;
            while (iter.hasNext()) {
                file = multiRequest.getFile(iter.next());
                if (file != null) {
                    String myFileName = file.getOriginalFilename();
                    if (myFileName.trim() != "") {
                        String fileName = "Upload" + file.getOriginalFilename();
                        String fileNameNew = Files.isExistName(imgPath, fileName);
                        path = imgPath + "\\" + fileNameNew;
                        stu.setStudentPic(fileNameNew);
                    }

                }
            }
            boolean isAdd = false;
            Student stu1 = new Student();
            if (stu != null) {
                stu1 = mInfoService.updateStuInfo(stu, imgPath);
                if (stu1 != null) {
                    isAdd = true;
                }
            }
            if (isAdd) {
                HttpSession session=request.getSession();
                session.removeAttribute("studentInfo");
                session.setAttribute("studentInfo", stu1);
                File localFile = new File(path);
                try {
                    if (file != null) {
                        file.transferTo(localFile);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                    msg += "\\\\\\\\\\\\" + e.getMessage();
                }
            }
        } else {
            String studentName = request.getParameter("studentName");
            String studentPassword = request.getParameter("studentPassword");
            String studentSex = request.getParameter("studentSex");
            String studentEmail = request.getParameter("studentEmail");
            Integer studentId = Integer.parseInt(request.getParameter("studentId"));
            String studentIntroduce = request.getParameter("studentIntroduce");
            Student stu = new Student();
            stu.setStudentEmail(studentEmail);
            stu.setStudentPassword(studentPassword);
            stu.setStudentIntroduce(studentIntroduce);
            stu.setStudentSex(studentSex);
            stu.setStudentName(studentName);
            stu.setStudentId(studentId);
            Student stu1 = mInfoService.updateStuInfo(stu);
            if (stu1 == null) {
                msg += "\\\\\\\\updateError";
            } else {
                HttpSession session=request.getSession();
                session.removeAttribute("studentInfo");
                session.setAttribute("studentInfo", stu1);
            }
        }
        if (msg.equals("")) {
            msg = "update success";
        }
        return "<script>alert('" + msg + "')</script>";
    }

    @RequestMapping(value = "updateTeacherInfo", method = RequestMethod.POST)
    @ResponseBody
    public Object updateTeachInfo(HttpServletRequest request) {
        String msg = "";
        if (mMultipartResolver.isMultipart(request)) {
            //转换成多部分request
            MultipartHttpServletRequest multiRequest = (MultipartHttpServletRequest) request;
            String teacherName = multiRequest.getParameter("teacherName");
            String teacherPassword = multiRequest.getParameter("teacherPassword");
            String teacherSex = multiRequest.getParameter("teacherSex");
            String teacherEmail = multiRequest.getParameter("email");
            Integer teacherId = Integer.parseInt(multiRequest.getParameter("teacherId"));
            String teacherIntroduce = multiRequest.getParameter("teacherIntroduce");
            Teacher t = new Teacher();
            t.setEmail(teacherEmail);
            t.setTeacherPassword(teacherPassword);
            t.setTeacherIntroduce(teacherIntroduce);
            t.setTeacherSex(teacherSex);
            t.setTeacherName(teacherName);
            t.setTeacherId(teacherId);
            Iterator<String> iter = multiRequest.getFileNames();
            String imgPath = request.getSession().getServletContext().getRealPath("/img");
            String path = "";
            MultipartFile file = null;
            while (iter.hasNext()) {
                file = multiRequest.getFile(iter.next());
                if (file != null) {
                    String myFileName = file.getOriginalFilename();
                    if (!myFileName.equals("")) {
                        String fileName = "Upload" + file.getOriginalFilename();
                        String fileNameNew = Files.isExistName(imgPath, fileName);
                        path = imgPath + "\\" + fileNameNew;
                        t.setTeacherPic(fileNameNew);
                    }

                }
            }
            boolean isAdd = false;
            Teacher t1 = new Teacher();
            if (t != null) {
                t1 = mInfoService.updateTeachInfo(t, imgPath);
                if (t1 != null)
                    isAdd = true;
            }
            if (isAdd) {
                HttpSession session=request.getSession();
                session.removeAttribute("teacherInfo");
                session.setAttribute("teacherInfo", t1);
                File localFile = new File(path);
                try {
                    if (file != null) {
                        file.transferTo(localFile);
                    }
                } catch (IOException e) {
                    msg += "\\\\\\\\\\\\" + e.getMessage();
                    e.printStackTrace();
                }
            }
        } else {
            String teacherName = request.getParameter("teacherName");
            String teacherPassword = request.getParameter("teacherPassword");
            String teacherSex = request.getParameter("teacherSex");
            String teacherEmail = request.getParameter("email");
            Integer teacherId = Integer.parseInt(request.getParameter("teacherId"));
            String teacherIntroduce = request.getParameter("teacherIntroduce");
            Teacher t = new Teacher();
            t.setEmail(teacherEmail);
            t.setTeacherPassword(teacherPassword);
            t.setTeacherIntroduce(teacherIntroduce);
            t.setTeacherSex(teacherSex);
            t.setTeacherName(teacherName);
            t.setTeacherId(teacherId);
            Teacher t1 = mInfoService.updateTeachInfo(t);
            if (t1 == null) {
                msg += "\\\\\\\\updateError";
            } else {
                HttpSession session=request.getSession();
                session.removeAttribute("teacherInfo");
                session.setAttribute("teacherInfo", t1);
            }
        }
        if (msg.equals("")) {
            msg = "update success";
        }
        return "<script>alert('" + msg + "')</script>";
    }
    @RequestMapping("faceLoginStuAndTeach")
    @ResponseBody
    public String stuOrTeachFaceLogin(HttpServletRequest request,String baseData,String userType) throws Exception {

        HttpSession session=request.getSession();
        if("student".equals(userType)) {
            Student stu1 = (Student) session.getAttribute("studentInfo");
            if (stu1 != null) {
                return "alreadyLogin!";
            } else {
               Student student = mLoginService.faceLoginStuBaidu(baseData);
                if (student != null) {
                    //写入session保存
                    System.out.println("StuLoginByFace pass,studentName:" + student.getStudentName());
                    HttpSession mysession = request.getSession();
                    mysession.setAttribute("studentInfo",  student);
                    return "index";
                } else {
                    System.out.println("forbidden student Login by face!");
                    return "fail";
                }
            }
        }else if("teacher".equals(userType)){
            Teacher tea1 = (Teacher) session.getAttribute("teacherInfo");
            if (tea1 != null) {
                return "alreadyLogin!";
            } else {
                Teacher teacher = mLoginService.faceLoginTeachBaidu(baseData);
                if (teacher != null) {
                    //写入session保存
                    System.out.println("TeacherLoginByFace pass,teacherName:" + teacher.getTeacherName());
                    HttpSession mysession = request.getSession();
                    mysession.setAttribute("teacherInfo",  teacher);
                    return "index";
                } else {
                    System.out.println("forbidden teacher Login by face!");
                    return "fail";
                }
            }
        }
        return "fail";
    }
}
