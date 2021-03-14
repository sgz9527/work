package com.xqq.service;

import com.xqq.dao.SchorcomMapper;
import com.xqq.dao.StudentMapper;
import com.xqq.dao.Teach_courseMapper;
import com.xqq.dao.TeacherMapper;
import com.xqq.pojo.SchOrCom;
import com.xqq.pojo.Student;
import com.xqq.pojo.Teacher;
import com.xqq.staticmethod.Files;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.util.HashMap;

@Service("InfoService")
public class InfoServiceImpl implements InfoService {
    @Resource
    StudentMapper studentDao;
    @Resource
    TeacherMapper teacherDao;
    @Resource
    SchorcomMapper schoolDao;
    @Resource
    Teach_courseMapper teach_courseDao;

    public Student updateStuInfo(Student student) {
        if (studentDao.updateByPrimaryKeySelective(student) == 1) {
            Student stu1 = studentDao.selectByPrimaryKey(student.getStudentId());
            return stu1;
        }
        return null;
    }

    public Student updateStuInfo(Student student, String imgPath) {
        String pic = studentDao.selectByPrimaryKey(student.getStudentId()).getStudentPic();
        String path = imgPath + "\\" + pic;
        if (Files.deleteFile1(path) != -1) {//删除之后
            if (studentDao.updateByPrimaryKeySelective(student) == 1) {
                Student stu1 = studentDao.selectByPrimaryKey(student.getStudentId());
                return stu1;
            }
        }

        return null;
    }

    public Teacher updateTeachInfo(Teacher teacher) {
        if (teacherDao.updateByPrimaryKeySelective(teacher) == 1) {
            Teacher teach1 = teacherDao.selectByPrimaryKey(teacher.getTeacherId());
            return teach1;
        }
        return null;
    }

    public Teacher updateTeachInfo(Teacher teacher, String imgPath) {
        String pic = teacherDao.selectByPrimaryKey(teacher.getTeacherId()).getTeacherPic();
        String path = imgPath + "\\" + pic;
        if (Files.deleteFile1(path) != -1) {
            if (teacherDao.updateByPrimaryKeySelective(teacher) == 1) {
                Teacher teach1 = teacherDao.selectByPrimaryKey(teacher.getTeacherId());
                return teach1;
            }
        }
        return null;
    }

    public HashMap<String, Object> getTeacherInfo(Integer teacherId) {
        Teacher teacher1 = teacherDao.selectByPrimaryKey(teacherId);
        return this.toHashMap(teacher1);
    }

    @Override
    public Student getStudent(Integer studentId) {
        return studentDao.selectByPrimaryKey(studentId);
    }

    public HashMap<String, Object> toHashMap(Teacher teacher) {
        HashMap<String, Object> map1 = new HashMap<String, Object>();
        Integer belongSchId = Integer.parseInt(teacher.getBelongSchId());
        SchOrCom sch = schoolDao.selectByPrimaryKey(belongSchId);
        map1.put("belongSchName", sch.getComName());
        int courseNum = teach_courseDao.selectByTeacherId(teacher.getTeacherId()).size();
        map1.put("courseNum", courseNum);
        return map1;
    }
}
