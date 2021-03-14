package com.xqq.dao;

import com.xqq.pojo.Student;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

@Repository("UserDao")
public interface IUserDao {
    public Student LoginStudent(@Param("StudentAccount") String StudentAccount, @Param("StudentPassword") String StudentPassword);

    public Student selectStudentById(@Param("studentId") Integer studentId);

    Student selectByAccount(String studentAccount);

}
