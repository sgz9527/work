package com.xqq.service;

import com.xqq.pojo.Manager;
import com.xqq.pojo.Student;
import com.xqq.pojo.Teacher;

import java.util.HashMap;

public interface LoginService {
    /**
     * 学生登录
     * @param StudentAccount
     * @param StudentPassword
     * @return
     */
    public Student LoginStudent(String StudentAccount, String StudentPassword);

    /**
     * 教师登录
     * @param teacherAccount
     * @param TeacherPassword
     * @return
     */

    public Teacher LoginTeacher(String teacherAccount, String TeacherPassword);

    /**
     * 管理员登录
     * @param account
     * @param password
     * @return
     */

    Manager loginAdmin(String account, String password);

    HashMap<String,Object> faceAdminLogin(String baseData)throws Exception;

    Manager faceAdminLoginBaidu(String baseData);

    String faceAdd(String baseData,String groupId,String userAccount);

    Student faceLoginStuBaidu(String baseData);

    Teacher faceLoginTeachBaidu(String baseData);

    int studentRegist(Student stu);

}
