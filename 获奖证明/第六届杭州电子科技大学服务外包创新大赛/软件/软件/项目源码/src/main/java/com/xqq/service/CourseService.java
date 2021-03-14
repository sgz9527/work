package com.xqq.service;

import com.xqq.pojo.*;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import javax.servlet.http.HttpSession;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public interface CourseService {
    /**
     * 获取首页资源展示课程
     * @return
     */
    List<HashMap<String, Object>> getIndexCourse();

    /**
     * 获取搜索课程信息
     * @param key
     * @return
     */

    public List<HashMap<String, Object>> getSearchCourse(String key);

    /**
     * 获取所有课程
     * @return
     */

    public List<HashMap<String, Object>> getAllCourse();

    /**
     * 将课程实体类集合转为json数组
     * @param courseList
     * @param num
     * @return
     */

    public ArrayList<HashMap<String, Object>> toHashMap(List<Course> courseList, int num);

    /**
     * 通过课程类型获取课程列表
     * @param type
     * @return
     */

    public List<HashMap<String, Object>> getCourseByType(String type);

    /**
     * 获取课程热度排行榜前十课程列表
     * @return
     */

    public List<HashMap<String, Object>> getTopTenCourse();

    /**
     * 通过课程ID获取指定课程信息
     * @param courseid
     * @param session
     * @return
     */

    public HashMap<String, Object> getUniqueCourse(Integer courseid, HttpSession session);

    /**
     * 得到指定课程ID的课程的授课教师列表
     * @param courseid
     * @return
     */

    public List<HashMap<String, Object>> getTeacher(Integer courseid);

    /**
     * 学生选课
     * @param courseid
     * @param session
     * @return
     */

    public boolean chooseCourse(Integer courseid,HttpSession session);

    /**
     * 获取学生已选课程
     * @param studentId
     * @return
     */

    public List<HashMap<String, Object>> getStudentCourse(Integer studentId);

    /**
     * 学生退选课程
     * @param courseId
     * @param studentId
     * @return
     */

    public int deleteCourse(Integer courseId, Integer studentId);

    /**
     * 获取为指定studentId的学生所推荐的课程列表
     * @param studentId
     * @return
     */

    public List<HashMap<String, Object>> getReCommandCourse(Integer studentId);

    /**
     * 获取指定课程ID的课程的所有公告
     * @param courseId
     * @return
     */

    public ArrayList<HashMap<String, Object>> getNotices(Integer courseId);

    /**
     * 获取指定课程ID的课程的所有章信息
     * @param courseId
     * @return
     */

    public ArrayList<HashMap<String, Object>> getChapters(Integer courseId);

    /**
     * 获取指定章ID的章的所有节信息
     * @param chapterId
     * @return
     */

    public ArrayList<HashMap<String, Object>> getUtils(Integer chapterId);

    /**
     * 获取指定UTILID的节的所有课件资源
     * @param utilId
     * @return
     */

    public ArrayList<HashMap<String, Object>> getSources(Integer utilId);

    /**
     * 增加课程资源信息
     * @param source
     * @return
     */

    public boolean addSource(Source source);

    /**
     * 获取指定课程ID的课程的所有测试信息
     * @param courseId
     * @param session
     * @return
     */

    public ArrayList<HashMap<String, Object>> getTest(Integer courseId,HttpSession session);

    /**
     * 获取指定课程ID的课程的所有待发布测试列表
     * @param courseId
     * @return
     */

    public ArrayList<HashMap<String, Object>> getWaitTest(Integer courseId);

    /**
     * 提交测试成绩
     * @param score
     * @return
     */

    public HashMap<String, Object> submitTest(Test_score score);

    /**
     * 提交测试基本信息，用于教师布置作业时
     * @param test
     * @return
     */

    public HashMap<String, Object> submitTestBase(Test test);

    /**
     * 详细布置测试的具体试卷内容
     * @param bigAndSmallList
     * @return
     */

    public HashMap<String, Object> submitDetailTest(ArrayList<testProblemAndSmall> bigAndSmallList);

    /**
     * 获取指定studentId的学生指定testId的test的成绩以及提交次数
     * @param studentId
     * @param testId
     * @return
     */

    public HashMap<String, Object> getScoreAndCount(Integer studentId, Integer testId);

    /**
     *获取成绩
     * @param studentId
     * @return
     */

    public ArrayList<HashMap<String, Object>> getScore(Integer studentId);

    /**
     *更新公告
     * @param notice
     * @return
     */

    public HashMap<String, Object> updateNotice(Notice notice);

    /**
     *新建公告并保存
     * @param notice
     * @return
     */

    public HashMap<String, Object> addSaveNotice(Notice notice);

    /**
     *发布公告
     * @param notice
     * @return
     */

    public HashMap<String, Object> proNotice(Notice notice);

    /**
     *删除保存但未发布的公告
     * @param noticeId
     * @return
     */

    public HashMap<String, Object> deleteNotice(Integer noticeId);

    /**
     *更新课程
     * @param course
     * @return
     */

    public HashMap<String, Object> updateCourse(Course course);

    /**
     *获取指定教师管理的课程列表
     * @param teacherId
     * @return
     */

    List<HashMap<String, Object>> getTeacherCourse(Integer teacherId);

    /**
     *添加课程申请审核信息
     * @param cc
     * @return
     */

    boolean addCheckCourse(CheckCourse cc);

    /**
     *获取指定教师提交的课程申请信息列表
     * @param teacherId
     * @return
     */

    List<HashMap<String, Object>> getTeacherCheckCourse(Integer teacherId);

    /**
     *添加章节信息
     * @param cauList
     * @return
     */

    String addChapterAndUtil(ArrayList<ChapterAndUtilList> cauList);

    /**
     *获取指定教师所教的课程下作业的所有成绩数据
     * @param teacherId
     * @return
     */

    List<HashMap<String, Object>> getScoreInfoByTeacherId(Integer teacherId);

    /**
     *获取指定学生的消息
     * @param studentId
     * @return
     */

    HashMap<String, Object> getMessageInfo(Integer studentId);

    /**
     *设置消息状态：已读/未读
     * @param messageIdList1
     * @param studentId
     * @return
     */

    boolean setMessageState(ArrayList<Integer> messageIdList1,Integer studentId);

    /**
     *检测所有作业的截止时间是否即将截止，如果即将截止则产生一条消息 提醒学生
     * @param studentId
     */

    void detectAllHomeWorkEndTime(Integer studentId);

    /**
     *删除消息
     */

    void deleteMessage();

    /**
     *用于管理员设置审核课程状态
     * @param checkCourse
     * @return
     */

    HashMap<String,Object> setCheckCourseState(CheckCourse checkCourse);

    JSONArray getAllCourseCheck();


    JSONObject getCourseAndExam(HttpSession session, Integer courseid);
    JSONObject getCourseAndExam(Integer studentId, Integer courseId);

    String submitExam(Integer studentId, Integer courseId, String one, String two, String three, String four, String five);

    String checkExam(Teach_courseKey tc);

    String deleteExam(Integer courseId);

    int submitExamInfo(Course course,HttpSession session);


    String submitStudentExamScore(Integer[] examScore, Integer studentId, Integer courseId);

    JSONArray getFinalExamScore(Integer studentId);

}