package com.xqq.service;

import com.xqq.pojo.Student;
import com.xqq.pojo.Teacher;

import java.util.HashMap;

public interface InfoService {
    /**
     * 更新学生个人信息，不指定图片路径
     * @param student
     * @return
     */
    public Student updateStuInfo(Student student);

    /**
     * 更新学生个人信息，指定图片路径
     * @param student
     * @param imgPath
     * @return
     */

    public Student updateStuInfo(Student student, String imgPath);

    /**
     *更新教师个人信息，指定图片路径
     * @param teacher
     * @return
     */

    public Teacher updateTeachInfo(Teacher teacher);

    /**
     *更新教师个人信息，不指定图片路径
     * @param teacher
     * @param imgPath
     * @return
     */

    public Teacher updateTeachInfo(Teacher teacher, String imgPath);

    /**
     *获取教师信息
     * @param teacherId
     * @return
     */

    public HashMap<String, Object> getTeacherInfo(Integer teacherId);

    /**
     *获取学生信息
     * @param studentId
     * @return
     */

    Student getStudent(Integer studentId);
}
