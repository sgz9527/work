package com.xqq.service;

import com.google.gson.Gson;
import com.xqq.constant.Constants;
import com.xqq.dao.*;
import com.xqq.pojo.*;
import com.xqq.staticmethod.GsonUtils;
import com.xqq.staticmethod.getDate;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.annotation.Resource;
import javax.servlet.http.HttpSession;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

/**
 * @author xqq
 */
@Service("courseService")
public class CourseServiceImpl implements CourseService {
    @Resource
    CourseMapper courseDao;
    @Resource
    Stu_courseMapper stu_course;
    @Resource
    Teach_courseMapper teach_courseDao;
    @Resource
    TeacherMapper teacherDao;
    @Resource
    NoticeMapper noticeDao;
    @Resource
    ChapterMapper chapterDao;
    @Resource
    UtilMapper utilDao;
    @Resource
    SourceMapper sourceDao;
    @Resource
    TestMapper testDao;
    @Resource
    TestProblemMapper testProblemDao;
    @Resource
    smallProblemMapper smallProblemDao;
    @Resource
    Test_scoreMapper test_scoreDao;
    @Resource
    SchorcomMapper schoolDao;
    @Resource
    CommentMapper commentDao;
    @Resource
    CheckCourseMapper mCheckCourseDao;
    @Resource
    StudentMapper studentDao;
    @Resource
    MessageMapper messageDao;
    @Resource
    TopicMapper topicDao;
    @Resource
    TrainingMapper trainDao;
    @Resource
    Message_stuMapper mMessageStuDao;
    @Resource
    private CommentMapper comM;
    @Resource
    private IUserDao userDao;

    public void addCourses() {
        for (int i = 0; i < 50; i++) {
            Course course = new Course();
            int choose = (int) (1 + Math.random() * 6);
            switch (choose) {
                case 1:
                    course.setCourseType("文科|外语");
                    break;
                case 2:
                    course.setCourseType("理科|地理|水土");
                    break;
                case 3:
                    course.setCourseType("工科|计算机|编程语言");
                    break;
                case 4:
                    course.setCourseType("文科|政治|思想理论");
                    break;
                case 5:
                    course.setCourseType("理科|数学|大学数学");
                    break;
                case 6:
                    course.setCourseType("工科|电子信息|电路基础");
                    break;
                default:break;
            }
            course.setBelongSchId(2);
            course.setCourseIntroduce("aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf");
            course.setCourseName("课程" + i);
            course.setCoursePostSrc("13.jpg");
            course.setIntroduceVideoSrc("video1.mp4");
            course.setEvaluationLevel("ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig");
            course.setCourseProgress(i);
            System.out.println(course.getCourseType());
            courseDao.insert(course);
        }
    }

    @Override
    public List<HashMap<String, Object>> getIndexCourse() {
        List<Course> courseList = courseDao.selectTop12();
        System.out.println("[service-getIndexCourse]courseList.size():" + courseList.size());
        return this.toHashMap(courseList, 12);
    }

    @Override
    public List<HashMap<String, Object>> getSearchCourse(String key) {
        List<Course> courseList = courseDao.selectByFuzzy(key);
        return this.toHashMap(courseList, courseList.size());

    }

    @Override
    public List<HashMap<String, Object>> getAllCourse() {
        List<Course> courseList = courseDao.selectAll();
        return this.toHashMap(courseList, courseList.size());

    }

    @Override
    public List<HashMap<String, Object>> getCourseByType(String type) {

        List<Course> courseList = courseDao.selectCourseByType(type);
        return this.toHashMap(courseList, courseList.size());
    }

    @Override
    public List<HashMap<String, Object>> getTopTenCourse() {
        List<Course> courseList = courseDao.selectAll();
        ArrayList<HashMap<String, Object>> mapLists = this.toHashMap(courseList, courseList.size());
        this.sort(mapLists, 0, courseList.size() - 1);
        return mapLists;
    }

    @Override
    public HashMap<String, Object> getUniqueCourse(Integer courseid,HttpSession session) {
        return this.toMap(courseDao.selectByPrimaryKey(courseid),session);
    }

    @Override
    public List<HashMap<String, Object>> getTeacher(Integer courseId) {
        List<Teach_courseKey> t_cList = teach_courseDao.selectByCourseId(courseId);
        ArrayList<Teacher> teacherList = new ArrayList<>();
        for (int i = 0; i < t_cList.size(); i++) {
            Teacher teacher = teacherDao.selectByPrimaryKey(t_cList.get(i).getTeacherId());
            teacherList.add(teacher);
        }
        return this.toHashMap1(teacherList, teacherList.size());
    }

    @Override
    public boolean chooseCourse(Integer courseId,HttpSession session) {
        Student student = (Student) session.getAttribute("studentInfo");
        if (student != null) {
            Stu_course stu_c = new Stu_course();
            stu_c.setCourseId(courseId);
            stu_c.setStudentId(student.getStudentId());
            stu_c.setChooseTime(getDate.getDate());
            if (stu_course.insert(stu_c) == 1) {
                return true;
            }
        }
        else {
            System.out.println("CourseServiceImpl.chooseCourse:student is null!");
        }
        return false;
    }

    @Override
    public List<HashMap<String, Object>> getStudentCourse(Integer studentId) {
        ArrayList<Course> courseList = new ArrayList<>();
        List<Stu_course> stu_cList = stu_course.selectByStudentId(studentId);
        System.out.println("CourseServiceImpl.getStudentCourse:已选课程:" + stu_cList.size() + "门");
        for (int i = 0; i < stu_cList.size(); i++) {
            Course course = courseDao.selectByPrimaryKey(stu_cList.get(i).getCourseId());
            courseList.add(course);
        }
        return this.toHashMap(courseList, courseList.size());
    }

    public ArrayList<Course> getStuCourselist(Integer studentId) {
        ArrayList<Course> courseList = new ArrayList<>();
        List<Stu_course> stu_cList = stu_course.selectByStudentId(studentId);
        System.out.println("CourseServiceImpl.getStuCourselist,已选课程:" + stu_cList.size() + "门");
        for (int i = 0; i < stu_cList.size(); i++) {
            Course course = courseDao.selectByPrimaryKey(stu_cList.get(i).getCourseId());
            courseList.add(course);
        }
        return courseList;
    }

    @Override
    @Transactional(rollbackFor=Exception.class)
    public int deleteCourse(Integer courseId, Integer studentId) {
        System.out.println("courseServImpl--deleteCourse,courseId:" + courseId + "  studentId:" + studentId);
        Stu_course stu_c = new Stu_course();
        stu_c.setCourseId(courseId);
        stu_c.setStudentId(studentId);
        int result1 = -1;
        String dateNow = getDate.getYMD();
        Stu_course stu_c1 = stu_course.selectByStudentIdAndCourseId(studentId, courseId);
        if (stu_c1 != null) {
            String chooseDate = getDate.getYMD(stu_c1.getChooseTime());
            System.out.println("chooseTime:" + chooseDate);
            ArrayList<String> nowList = segment(dateNow, '-');
            ArrayList<String> endList = segment(chooseDate, '-');
            int nowYear = Integer.parseInt(nowList.get(0));
            int chooseYear = Integer.parseInt(endList.get(0));
            int nowMonth = Integer.parseInt(nowList.get(1));
            int chooseMonth = Integer.parseInt(endList.get(1));
            int nowDay = Integer.parseInt(nowList.get(2));
            int chooseDay = Integer.parseInt(endList.get(2));
            if (nowYear == chooseYear && nowMonth == chooseMonth && (nowDay - chooseDay) <= 3) {
                //只有三天之内可以删除
                result1 = stu_course.deleteByPrimaryKey(stu_c);
                ArrayList<Test_score> testScoreList = test_scoreDao.selectByStudentIdAndCourseId(studentId, courseId);
                for (int i = 0; i < testScoreList.size(); i++) {
                    test_scoreDao.deleteByPrimaryKey(testScoreList.get(i).getScoreId());
                }
            }
            //2表示超过三天，不能删除
            else {
                result1 = 2;
            }
        }
        return result1;
    }

    @Override
    public List<HashMap<String, Object>> getReCommandCourse(Integer studentId) {
        System.out.println("CourseServiceImpl.getReCommandCourse");
        List<Course> courseList;
        //得到课程列表
        courseList = this.getStuCourselist(studentId);
        //一门课程多个标签集合的集合
        List<ArrayList<String>> courseTypeLists = new ArrayList<ArrayList<String>>();
        for (int i = 0; i < courseList.size(); i++) {
            ArrayList<String> typeStrList = segment(courseList.get(i).getCourseType(), '|');
            //将一门课程的类型标签集合加入到courseTypeLists中
            courseTypeLists.add(typeStrList);
        }
        //所有课程第一层标签构成的集合
        ArrayList<String> strListFirst = new ArrayList<>();
        for (int i = 0; i < courseTypeLists.size(); i++) {
            if(courseTypeLists.get(i).size()!=0){
                strListFirst.add(courseTypeLists.get(i).get(0));
            }else{
                strListFirst.add("其他");
            }
        }
        //得到第一层中数量最多的标签
        String firstLabel = getBiggest(strListFirst);
        List<Course> resultList = courseDao.selectCourseByType(firstLabel);
        //得到所有未选的推荐课程
        ArrayList<Course> resultCourseList=new ArrayList<>();
        for(int i=0;i<resultList.size();i++){
            boolean isChoose=false;
            for(int j=0;j<courseList.size();j++){
                //如果已选过，则在结果推荐列表中删除已选
                if(courseList.get(j).getCourseId().equals(resultList.get(i).getCourseId())){
                    isChoose=true;
                    break;
                }
            }
            if(!isChoose){
                resultCourseList.add(resultList.get(i));
            }
        }
        //得到第二层标签集合
        ArrayList<String> secondListFirst = new ArrayList<>();
        for (int i = 0; i < courseTypeLists.size(); i++) {
            if(firstLabel.equals(courseTypeLists.get(i).get(0))) {
                if (courseTypeLists.get(i).size() >= 2) {
                    secondListFirst.add(courseTypeLists.get(i).get(1));
                } else {
                    secondListFirst.add("其他");
                }
            }
        }
        //得到第二层数量最多的标签
        String secondLabel = getBiggest(secondListFirst);
        String courseTypeFS=firstLabel+"|"+secondLabel;
        //得到根据第1,2层标签搜索推荐的结果课程列表
        List<Course> resultList1 = courseDao.selectCourseByType(courseTypeFS);
        //如果不在resultCourseList中，则从resultCourseList1中剔除(说明是用户已选过的课程)
        ArrayList<Course> resultCourseList1=new ArrayList<>();
        for(int i=0;i<resultList1.size();i++){
            //默认不存在的标志信号
            boolean isExistInRes=false;
            for(int j=0;j<resultCourseList.size();j++){
                if(resultCourseList.get(j).getCourseId().equals(resultList1.get(i).getCourseId())){
                    //如果在resultCourseList中，置标志为true
                    isExistInRes=true;
                    break;
                }
            }
            if(isExistInRes){
                //存在即加入之
                resultCourseList1.add(resultList1.get(i));
            }
        }
        ArrayList<String> thirdListFirst = new ArrayList<>();
        for (int i = 0; i < courseTypeLists.size(); i++) {
            if (secondLabel.equals(courseTypeLists.get(i).get(1))) {
                if (courseTypeLists.get(i).size() >= 3) {
                    thirdListFirst.add(courseTypeLists.get(i).get(2));
                } else {
                    thirdListFirst.add("其他");
                }
            }
        }
        //得到第三层数量最多的标签
        String thirdLabel = getBiggest(thirdListFirst);
        String courseTypeFST=firstLabel+"|"+secondLabel+"|"+thirdLabel;
        //得到根据第1,2,3层标签搜索推荐的结果课程列表
        List<Course> resultList2 = courseDao.selectCourseByType(courseTypeFST);
        //如果不在resultCourseList1中，则从resultCourseList2中剔除(说明是用户已选过的课程)
        ArrayList<Course> resultCourseList2=new ArrayList<>();
        for(int i=0;i<resultList2.size();i++){
            //默认不存在的标志信号
            boolean isExistInRes=false;
            for(int j=0;j<resultCourseList1.size();j++){
                if(resultList2.get(i).getCourseId().equals(resultCourseList1.get(j).getCourseId())){
                    //如果在resultCourseList中，置标志为true
                    isExistInRes=true;
                    break;
                }
            }
            if(isExistInRes){
                resultCourseList2.add(resultList2.get(i));
            }
        }
        return this.toHashMap(resultCourseList2, resultCourseList2.size());
    }

    @Override
     /**
     noticeState == "finish") {
     ["noticeTitle"] ["noticeBody"] ["noticeWritter"]  ["noticeTime"]
    ["noticeSystemTime"]
     */
    public ArrayList<HashMap<String, Object>> getNotices(Integer courseId) {
        ArrayList<Notice> noticeList = noticeDao.selectByCourseId(courseId);
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        for (int i = 0; i < noticeList.size(); i++) {
            HashMap<String, Object> map = new HashMap<>(7);
            map.put("noticeState", noticeList.get(i).getNoticeState());
            map.put("noticeTitle", noticeList.get(i).getNoticeTitle());
            map.put("noticeBody", noticeList.get(i).getNoticeDetail());
            map.put("noticeWritter", noticeList.get(i).getWriter());
            map.put("noticeTime", noticeList.get(i).getTime());
            map.put("noticeSystemTime", getDate.getHMS(noticeList.get(i).getSystemTime()));
            map.put("noticeId", noticeList.get(i).getNoticeId());
            mapList.add(map);
        }
        return mapList;
    }

    @Override
    /**
     * ["chapterId"]
     * ["chapterId"]
     * ["chapterId"]  ["titleNumberName"]["titleName"]
     * ["chapterId"]
     */
    public ArrayList<HashMap<String, Object>> getChapters(Integer courseId) {
        ArrayList<Chapter> chapterList = chapterDao.selectByCourseId(courseId);
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        for (int i = 0; i < chapterList.size(); i++) {
            HashMap<String, Object> map = new HashMap<>(3);
            map.put("chapterId", chapterList.get(i).getChapterId());
            map.put("titleNumberName", chapterList.get(i).getChapterName());
            map.put("titleName", chapterList.get(i).getChapterTitle());
            mapList.add(map);
        }
        return mapList;
    }

    @Override
    /**
     * utilId   utilDetailTitle  utilDetailName+
     * utilDetailVideoNumber
     * utilDetailTextNumber
     */
    public ArrayList<HashMap<String, Object>> getUtils(Integer chapterId) {
        ArrayList<Util> utilList = utilDao.selectByChapterId(chapterId);
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();

        for (int i = 0; i < utilList.size(); i++) {
            HashMap<String, Object> map = new HashMap<>(5);
            map.put("utilId", utilList.get(i).getUtilId());
            map.put("utilDetailTitle", utilList.get(i).getUtilTitle());
            map.put("utilDetailName", utilList.get(i).getUtilName());
            map.put("utilDetailVideoNumber", sourceDao.selectByUtilIdAndType(utilList.get(i).getUtilId(), "video").size());
            map.put("utilDetailTextNumber", sourceDao.selectByUtilIdAndType(utilList.get(i).getUtilId(), "text").size());
            mapList.add(map);
        }
        return mapList;
    }

    @Override
    /**
     * hashMap key:
     * videoposter
     * sourceSrc
     * sourceType
     */
    public ArrayList<HashMap<String, Object>> getSources(Integer utilId) {
        ArrayList<Source> sourceList = sourceDao.selectByUtilId(utilId);
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<HashMap<String, Object>>();
        for (int i = 0; i < sourceList.size(); i++) {
            HashMap<String, Object> map = new HashMap<>(3);
            map.put("videoposter", sourceList.get(i).getPoster());
            map.put("sourceSrc", sourceList.get(i).getSourceSrc());
            map.put("sourceType", sourceList.get(i).getSourceType());
            mapList.add(map);
        }
        return mapList;
    }

    @Override
    /**
     * 每当教师上传一个章节的资源时，则置课程进度为本单元，并产生一条消息记录提醒课程进度
     */
    @Transactional(rollbackFor = Exception.class)
    public boolean addSource(Source source) {
        int res = sourceDao.insert(source);
        if (res == 1) {
            Message msg = new Message();
            msg.setMessageType(Constants.COURSE_OPEN);
            Util util=utilDao.selectByPrimaryKey(source.getBelongUtilId());
            int chapterId=util.getBelongChapterId();
            Chapter chapter=chapterDao.selectByPrimaryKey(chapterId);
            int courseId=chapter.getBelongCourseId();
            int chapterOrder=chapter.getChapterOrder();
            Course course=new Course();
            course.setCourseId(courseId);
            course.setCourseProgress(chapterOrder);
            //更新本课程开课进度
            courseDao.updateByPrimaryKeySelective(course);
            msg.setMessageCourseId(courseId);
            msg.setMessageTime(new Date());
            //产生一条消息
            messageDao.insertSelective(msg);
            return true;
        }
        return false;
    }
    /**
     * test是否截止
     */
    public boolean isStop(String dateNow, String dateEnd) {
        ArrayList<String> nowList = segment(dateNow, '-');
        ArrayList<String> endList = segment(dateEnd, '-');
        int nowYear = Integer.parseInt(nowList.get(0));
        int endYear = Integer.parseInt(endList.get(0));
        int nowMonth = Integer.parseInt(nowList.get(1));
        int endMonth = Integer.parseInt(endList.get(1));
        int nowDay = Integer.parseInt(nowList.get(2));
        int endDay = Integer.parseInt(endList.get(2));
        if (nowYear > endYear) {
            return true;
        } else if (nowYear < endYear) {
            return false;
        } else {
            if (nowMonth > endMonth) {
                return true;
            } else if (nowMonth < endMonth) {
                return false;
            } else {
                if (nowDay > endDay) {
                    return true;
                } else {
                    return false;
                }
            }
        }
    }

    @Override
    public ArrayList<HashMap<String, Object>> getTest(Integer courseId,HttpSession session) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        //章节集合
        ArrayList<Chapter> chapterList;
        chapterList = chapterDao.selectByCourseId(courseId);
        System.out.println("courseServiceImpl--getTest--chapterList.size():" + chapterList.size());
        for (int i = 0; i < chapterList.size(); i++) {
            ArrayList<Test> testList;
            //得到每一章的作业集合
            testList = testDao.selectByChapterId(chapterList.get(i).getChapterId());
            for (int j = 0; j < testList.size(); j++) {
                HashMap<String, Object> map1 = new HashMap<>(15);
                map1.put("chapterId", chapterList.get(i).getChapterId());
                map1.put("chapterName", chapterList.get(i).getChapterName());
                map1.put("testId", testList.get(j).getTestId());
                map1.put("testName", testList.get(j).getTestName());
                map1.put("testStartTime", testList.get(j).getTestStartTime());
                map1.put("testAllMark", testList.get(j).getTestAllMark());
                map1.put("submitCount", testList.get(j).getSubmitCount());
                String state = "error";
                Date dateNow = new Date();
                Date endTime = testList.get(j).getTestEndTime();
                SimpleDateFormat dt = new SimpleDateFormat("yyyy-MM-dd");
                //包含年月日的日期字符串
                String dateN = dt.format(dateNow);
                String endD = dt.format(endTime);
                map1.put("testEndTime", endD);
                if (this.isStop(dateN, endD)) {
                    state = "已截止";
                } else {
                    Student stu = (Student) session.getAttribute("studentInfo");
                    if (stu != null) {
                        Test_score tesScore = test_scoreDao.selectByStudentIdAndTestId(stu.getStudentId(), testList.get(j).getTestId());
                        //存在着条记录，说明做过
                        if (tesScore != null)
                        {
                            state = "已完成";
                        } else {
                            state = "待做";
                        }
                    }
                }
                map1.put("testState", state);
                map1.put("testType", testList.get(j).getTestType());
                map1.put("limitTime", testList.get(j).getLimitTime());
                map1.put("testIntro", testList.get(j).getTestIntro());
                ArrayList<HashMap<String, Object>> mapList1 = this.getTestDetail(testList.get(j).getTestId());
                //HashMap<String,Object> map2=new HashMap<String, Object>();
                // ArrayList<TestProblem> testProblemList=testProblemDao.selectByTestId(testList.get(j).getTestId());//获取指定TestId的大题集合
                //将试卷详细信息放进去(mapList1即为试卷详细信息)
                map1.put("testDetail", mapList1);
                //放入最外层的ArrayList中
                mapList.add(map1);
            }
        }

        return mapList;
    }

    @Override
    public ArrayList<HashMap<String, Object>> getWaitTest(Integer courseId) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        //章节集合
        ArrayList<Chapter> chapterList;
        chapterList = chapterDao.selectByCourseId(courseId);
        System.out.println("courseServiceImpl--getWaitTest--chapterList.size():" + chapterList.size());
        for (int i = 0; i < chapterList.size(); i++) {
            ArrayList<Test> testList;
            //得到每一章的作业集合
            testList = testDao.selectByChapterIdWait(chapterList.get(i).getChapterId());
            for (int j = 0; j < testList.size(); j++) {
                HashMap<String, Object> map1 = new HashMap<>(15);
                map1.put("chapterId", chapterList.get(i).getChapterId());
                map1.put("chapterName", chapterList.get(i).getChapterName());
                map1.put("testId", testList.get(j).getTestId());
                map1.put("testName", testList.get(j).getTestName());
                map1.put("testStartTime", testList.get(j).getTestStartTime());
                map1.put("testAllMark", testList.get(j).getTestAllMark());
                map1.put("submitCount", testList.get(j).getSubmitCount());
                Date dateNow = new Date();
                Date endTime = testList.get(j).getTestEndTime();
                SimpleDateFormat dt = new SimpleDateFormat("yyyy-MM-dd");
                //包含年月日的日期字符串
                String dateN = dt.format(dateNow);
                String endD = dt.format(endTime);
                map1.put("testEndTime", endD);
                map1.put("testType", testList.get(j).getTestType());
                map1.put("limitTime", testList.get(j).getLimitTime());
                map1.put("testIntro", testList.get(j).getTestIntro());
                //放入最外层的ArrayList中
                mapList.add(map1);
            }
        }

        return mapList;
    }

    @Override
    public HashMap<String, Object> submitTest(Test_score score) {
        int result1 = 0;
        int result2 = 0;
        HashMap<String, Object> map = new HashMap<>(15);
        Test_score tes = test_scoreDao.selectByStudentIdAndTestId(score.getStudentId(), score.getTestId());
        //如果数据表中有这条记录
        if (tes != null)
        {
            //如果上一次的分数比这次低，则更新成绩
            if (tes.getScore() < score.getScore())
            {
                result1 = test_scoreDao.updateScore(score.getScore(), score.getStudentId(), score.getTestId());
            } else {
                result1 = 1;
            }
        } else//没有的话就插入一条
        {
            result2 = test_scoreDao.insertSelective(score);
        }
        int count = -1;
        //接下来更新提交次数
        if (result1 == 1 || result2 == 1) {
            test_scoreDao.updateAlreadyCount(score.getStudentId(), score.getTestId());
            count = test_scoreDao.selectByStudentIdAndTestId(score.getStudentId(), score.getTestId()).getAlreadyCount();
        }
        map.put("submitCount", count);
        return map;
    }

    @Override
    public HashMap<String, Object> submitTestBase(Test test) {
        HashMap<String, Object> map = new HashMap<String, Object>();
        int res = testDao.insertSelective(test);
        if (res == 1) {
            map.put("result", "ok");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public HashMap<String, Object> submitDetailTest(ArrayList<testProblemAndSmall> bigAndSmallList) {
        Boolean isSuccess = true;
        Integer testId=-1;
        if(bigAndSmallList.size()!=0){
            testId=bigAndSmallList.get(0).getTestProblem().getBelongTestId();
        }
        for (int i = 0; i < bigAndSmallList.size(); i++) {
            //插入大题记录后返回其id
            Integer res = testProblemDao.insertSelective(bigAndSmallList.get(i).getTestProblem());
            if (res != 0) {
                ArrayList<smallProblem> smallList = (ArrayList<smallProblem>) bigAndSmallList.get(i).getSmallProblemList();
                for (int j = 0; j < smallList.size(); j++) {
                    smallList.get(j).setBelongTestProblemId(bigAndSmallList.get(i).getTestProblem().getTestProblemId());
                    //插入小题记录
                    int ss = smallProblemDao.insertSelective(smallList.get(j));
                    if (ss != 1) {
                        isSuccess = false;
                        break;
                    }
                }
            } else {
                isSuccess = false;
                break;
            }
        }
        Test test = new Test();
        test.setTestId(bigAndSmallList.get(0).getTestProblem().getBelongTestId());
        test.setTestState((short) 1);
        int s1 = testDao.updateByPrimaryKeySelective(test);
        if (s1 != 1) {
            isSuccess = false;
        }
        HashMap<String, Object> map = new HashMap<>(1);
        if (isSuccess) {
            Message msg=new Message();
            msg.setMessageTestId(testId);
            msg.setMessageCourseId(chapterDao.selectByPrimaryKey(testDao.selectByPrimaryKey(testId).getBelongChapterId()).getBelongCourseId());
            msg.setMessageType( Constants.COURSE_HOMEWORK_PRO);
            msg.setMessageTime(new Date());
            messageDao.insertSelective(msg);
            map.put("result", "ok");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @Override
    public HashMap<String, Object> getScoreAndCount(Integer studentId, Integer testId) {
        int count = -1;
        float score = -1;
        HashMap<String, Object> map = new HashMap<>(2);
        Test_score tes = test_scoreDao.selectByStudentIdAndTestId(studentId, testId);
        //如果有这条记录
        if (tes != null) {
            count = tes.getAlreadyCount();
            score = tes.getScore();
        }
        map.put("submitCount", count);
        map.put("score", score);
        return map;
    }

    @Override
    public ArrayList<HashMap<String, Object>> getScore(Integer studentId) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        ArrayList<Stu_course> stu_cList = (ArrayList<Stu_course>) stu_course.selectByStudentId(studentId);
        for (int i = 0; i < stu_cList.size(); i++) {
            ArrayList<Test_score> test_scoreArry = test_scoreDao.selectByStudentIdAndCourseId(studentId, stu_cList.get(i).getCourseId());
            //一门课程对应一个期末测试
            if(!stu_cList.get(i).getExamScore().equals(Constants.NOT_FINISH_EXAM)&&!stu_cList.get(i).getExamScore().equals(Constants.FINISH_NOT_READ)){
                HashMap<String,Object> map2=new HashMap<>(5);
                map2.put("courseName",courseDao.selectByPrimaryKey(stu_cList.get(i).getCourseId()).getCourseName());
                map2.put("courseId",stu_cList.get(i).getCourseId());
                map2.put("scoreType", "考试");
                map2.put("score", stu_cList.get(i).getExamScore());
                map2.put("scoreTitle",courseDao.selectByPrimaryKey(stu_cList.get(i).getCourseId()).getCourseName()+"期末考试" );
                mapList.add(map2);
            }
            //该名学生在该门课程上有测验成绩，类型为测试/作业
            for (int j = 0; j < test_scoreArry.size(); j++) {
                HashMap<String, Object> map1 = new HashMap<>(5);
                map1.put("courseName", courseDao.selectByPrimaryKey(stu_cList.get(i).getCourseId()).getCourseName());
                map1.put("courseId", stu_cList.get(i).getCourseId());
                map1.put("scoreType", "测试/作业");
                map1.put("score", test_scoreArry.get(j).getScore());
                map1.put("scoreTitle", testDao.selectByPrimaryKey(test_scoreArry.get(j).getTestId()).getTestName());
                mapList.add(map1);
            }

        }
        return mapList;
    }

    @Override
    public HashMap<String, Object> updateNotice(Notice notice) {
        HashMap<String, Object> map = new HashMap<>(1);
        if (noticeDao.updateByPrimaryKey(notice) == 1) {
            map.put("result", "ok");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @Override
    public HashMap<String, Object> addSaveNotice(Notice notice) {
        HashMap<String, Object> map = new HashMap<>(1);
        if (noticeDao.insert(notice) == 1) {
            map.put("result", notice.getNoticeId());
        } else {
            map.put("result", -1);
        }
        return map;
    }

    @Override
    public HashMap<String, Object> proNotice(Notice notice) {
        HashMap<String, Object> map = new HashMap<>(1);
        if (noticeDao.updateByPrimaryKeySelective(notice) == 1) {
            map.put("result", "ok");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @Override
    public HashMap<String, Object> deleteNotice(Integer noticeId) {
        HashMap<String, Object> map = new HashMap<>(1);
        if (noticeDao.deleteByPrimaryKey(noticeId) == 1) {
            map.put("result", "ok");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @Override
    public HashMap<String, Object> updateCourse(Course course) {
        HashMap<String, Object> map = new HashMap<>(1);
        if (courseDao.updateByPrimaryKeySelective(course) == 1) {
            map.put("result", "提交成功");
        } else {
            map.put("result", "提交失败");
        }
        return map;
    }

    @Override
    public List<HashMap<String, Object>> getTeacherCourse(Integer teacherId) {
        ArrayList<Course> courseList = new ArrayList<>();
        ArrayList<Teach_courseKey> teacList = teach_courseDao.selectByTeacherId(teacherId);
        for (int i = 0; i < teacList.size(); i++) {
            courseList.add(courseDao.selectByPrimaryKey(teacList.get(i).getCourseId()));
        }
        return this.toHashMap(courseList, courseList.size());
    }

    @Override
    public boolean addCheckCourse(CheckCourse cc) {
        if (mCheckCourseDao.insertSelective(cc) == 1) {
            return true;
        }
        return false;
    }

    @Override
    public List<HashMap<String, Object>> getTeacherCheckCourse(Integer teacherId) {
        ArrayList<CheckCourse> ccList;
        ccList = mCheckCourseDao.selectByTeacherId(teacherId);
        return this.toMapList(ccList);
    }

    @Override
    @Transactional
    public String addChapterAndUtil(ArrayList<ChapterAndUtilList> cauList) {
        String errorMsg = "";
        if (cauList.size() != 0) {
            //获取checkId
            Integer checkId = cauList.get(0).getChapter().getBelongCourseId();
            CheckCourse checkCourse = mCheckCourseDao.selectByPrimaryKey(checkId);
            //删除审核记录成功后执行插入课程，章节记录
            if (mCheckCourseDao.deleteByPrimaryKey(checkId) == 1)
            {
                Course course = new Course();
                course.setCourseProgress(0);
                course.setIntroduceVideoSrc(checkCourse.getIntroduceSrc());
                course.setCourseName(checkCourse.getCourseName());
                course.setCourseType(checkCourse.getCourseType());
                course.setBelongSchId(checkCourse.getBelongSchId());
                course.setCourseIntroduce(checkCourse.getCourseIntroduce());
                course.setCoursePostSrc(checkCourse.getPosterSrc());
                if (courseDao.insertSelective(course) == 1) {
                    Teach_courseKey teach_c = new Teach_courseKey();
                    teach_c.setCourseId(course.getCourseId());
                    teach_c.setTeacherId(checkCourse.getTeacherId());
                    teach_courseDao.insert(teach_c);
                    for (int i = 0; i < cauList.size(); i++) {
                        cauList.get(i).getChapter().setBelongCourseId(course.getCourseId());
                        if (chapterDao.insertSelective(cauList.get(i).getChapter()) == 1) {
                            for (int j = 0; j < cauList.get(i).getUtilList().size(); j++) {
                                cauList.get(i).getUtilList().get(j).setBelongChapterId(cauList.get(i).getChapter().getChapterId());
                                if (utilDao.insertSelective(cauList.get(i).getUtilList().get(j)) != 1) {
                                    errorMsg += "\\\\insert util error";
                                }
                            }
                        } else {
                            errorMsg += "\\\\insert to chapter error";
                        }
                    }
                } else {
                    errorMsg += "\\\\insert to course error";
                }
            } else {
                errorMsg += "\\\\\\delete check document error";
            }
        } else {
            errorMsg += "\\\\null list";
        }
        return errorMsg;
    }

    @Override
    public List<HashMap<String, Object>> getScoreInfoByTeacherId(Integer teacherId) {
        Integer courseId;
        ArrayList<Teach_courseKey> tcList = teach_courseDao.selectByTeacherId(teacherId);
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        for (int i = 0; i < tcList.size(); i++) {
            courseId = tcList.get(i).getCourseId();
            HashMap<String, Object> map = new HashMap<>();
            Course course1 = courseDao.selectByPrimaryKey(courseId);
            ArrayList<Stu_course> examScoreList=(ArrayList<Stu_course>)stu_course.selectByCourseId(courseId);
            JSONArray jsar=new JSONArray();
            for(int n=0;n<examScoreList.size();n++){
                if(!examScoreList.get(n).getExamScore().equals(Constants.FINISH_NOT_READ)&&!examScoreList.get(n).getExamScore().equals(Constants.NOT_FINISH_EXAM)){
                    StuCourse stuc=new StuCourse(examScoreList.get(n));
                    Student stu=studentDao.selectByPrimaryKey(examScoreList.get(n).getStudentId());
                    stuc.setStudentName(stu.getStudentName());
                    stuc.setStudentAccount(stu.getStudentAccount());
                    stuc.setCourseName(courseDao.selectByPrimaryKey(examScoreList.get(n).getCourseId()).getCourseName());
                    String jstr= GsonUtils.toJson(stuc);
                    JSONObject jsob=null;
                    try {
                        jsob=(JSONObject)new JSONParser().parse(jstr);
                    } catch (ParseException e) {
                        e.printStackTrace();
                    }
                    jsar.add(jsob);
                }
            }
            map.put("courseName", course1.getCourseName());
            map.put("courseId", course1.getCourseId());
            map.put("examList",jsar);
            map.put("passPerc", this.getPassPerc1(jsar) + "%");
            map.put("goodPerc",this.getGoodPerc1(jsar)+ "%");
            ArrayList<Chapter> chapterList1 = chapterDao.selectByCourseId(courseId);
            ArrayList<HashMap<String, Object>> testMapList = new ArrayList<>();
            for (int j = 0; j < chapterList1.size(); j++) {
                ArrayList<Test> testList1 = testDao.selectByChapterId(chapterList1.get(j).getChapterId());
                for (int k = 0; k < testList1.size(); k++) {
                    HashMap<String, Object> testMap = new HashMap<>(15);
                    testMap.put("testOrExamName", testList1.get(k).getTestName());
                    ArrayList<Test_score> tsList = test_scoreDao.selectByTestId(testList1.get(k).getTestId());
                    ArrayList<HashMap<String, Object>> scoreMapList = new ArrayList<HashMap<String, Object>>();
                    for (int t = 0; t < tsList.size(); t++) {
                        HashMap<String, Object> scoreMap = new HashMap<>();
                        Student student = studentDao.selectByPrimaryKey(tsList.get(t).getStudentId());
                        scoreMap.put("scoreBelongName", student.getStudentName());
                        scoreMap.put("scoreBelongAccount", student.getStudentAccount());
                        scoreMap.put("scoreMark", tsList.get(t).getScore());
                        scoreMapList.add(scoreMap);
                    }
                    testMap.put("passPerc", this.getPassPerc(tsList) + "%");
                    testMap.put("goodPerc", this.getGoodPerc(tsList) + "%");
                    testMap.put("scoreList", scoreMapList);
                    testMapList.add(testMap);
                }
                map.put("testOrExamList", testMapList);
            }
            mapList.add(map);
        }
        return mapList;
    }

    private Double getPassPerc1(JSONArray jsar) {
        int count = 0;
        for (int i = 0; i < jsar.size(); i++) {
            JSONObject jsob=(JSONObject)jsar.get(i);
            Double score=Double.parseDouble(jsob.get("allScore").toString());
            if(score>=60){
                count++;
            }
        }
        if (jsar.size() != 0) {
            return (count / ((double) jsar.size())) * 100;
        }
        return 0.0;

    }
    private Double getGoodPerc1(JSONArray jsar) {
        int count = 0;
        for (int i = 0; i < jsar.size(); i++) {
            JSONObject jsob=(JSONObject)jsar.get(i);
            Double score=Double.parseDouble(jsob.get("allScore").toString());
            if(score>=80){
                count++;
            }
        }
        if (jsar.size() != 0) {
            return (count / ((double) jsar.size())) * 100;
        }
        return 0.0;

    }

    @Override
    /** 1.课程开课进度
        2.提醒课程作业发布
        3.课程考试发布
        4.评论回复提醒
        5.帖子有新评论提醒
        6.作业截止提醒
        7.考试截止提醒
        8.实训发布提醒
        9.评论禁言提醒
        10.禁止发帖提醒
     */
    public HashMap<String, Object> getMessageInfo(Integer studentId) {
        ArrayList<Message> messageList = messageDao.selectByStudentId(studentId);
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        HashMap<String, Object> map1 = new HashMap<>(20);
        System.out.println("CourseServiceImpl.getMessageInfo:mapList.size():"+messageList.size());
        for (int i = 0; i < messageList.size(); i++) {
            HashMap<String, Object> map = new HashMap<>(20);
            map.put("messageType", messageList.get(i).getMessageType());
            if(mMessageStuDao.selectByPrimaryKey(messageList.get(i).getMessageId(),studentId)!=null){
                map.put("messageState", 1);
            }
            else{
                map.put("messageState", -1);
            }
            switch (messageList.get(i).getMessageType()) {
                case Constants.COURSE_OPEN:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    Course cour=courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId());
                    map.put("courseName", cour.getCourseName());
                    map.put("courseProgress",cour.getCourseProgress());
                    break;
                case Constants.COURSE_HOMEWORK_PRO:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    map.put("courseName", courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId()).getCourseName());
                    map.put("testName", testDao.selectByPrimaryKey(messageList.get(i).getMessageTestId()).getTestName());
                    break;
                case Constants.COURSE_EXAM_PRO:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    map.put("courseName", courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId()).getCourseName());
                    Course course=courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId());
                    map.put("examName", course.getCourseName()+"期末测试");
                    break;
                case Constants.COMMENT_REPLY:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    map.put("courseName", courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId()).getCourseName());
                    map.put("topicName", topicDao.selectByPrimaryKey(messageList.get(i).getMessageTopicId()).getTopicDetail());
                    map.put("replyInfo", commentDao.selectByPrimaryKey1(messageList.get(i).getMessageCommentId()).getCommentInfo());
                    Integer replyToId = commentDao.selectByPrimaryKey1(messageList.get(i).getMessageCommentId()).getReplyToId();
                    String commentInfo = commentDao.selectByPrimaryKey1(replyToId).getCommentInfo();
                    map.put("commentInfo", commentInfo);
                    break;
                case Constants.TOPIC_NEW_COMMENTS:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    map.put("courseName", courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId()).getCourseName());
                    map.put("topicName", topicDao.selectByPrimaryKey(messageList.get(i).getMessageTopicId()).getTopicDetail());
                    map.put("commentInfo", commentDao.selectByPrimaryKey1(messageList.get(i).getMessageCommentId()).getCommentInfo());
                    break;
                case Constants.HOMEWORK_END:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    map.put("courseName", courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId()).getCourseName());
                    Test test = testDao.selectByPrimaryKey(messageList.get(i).getMessageTestId());
                    map.put("testName", test.getTestName());
                    map.put("testEndTime", getDate.getYMD(test.getTestEndTime()));
                    break;
                case Constants.EXAM_END:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    Course course1=courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId());
                    map.put("courseName",course1.getCourseName());
                    map.put("examName", course1.getCourseName()+"期末测试");
                    map.put("examEndTime", getDate.getYMD(course1.getExamEndTime()));
                    break;
                case Constants.COMMENT_FORBIDEN:
                    System.out.println("CourseServiceImpl.getMessageInfo,commentForbiden:");
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    map.put("courseName", courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId()).getCourseName());
                    map.put("forbidenBeginTime", getDate.getYMD(messageList.get(i).getMessageForbidenBeginTime()));
                    map.put("forbidenEndTime", getDate.getYMD(messageList.get(i).getMessageForbidenEndTime()));
                    map.put("commentInfo", commentDao.selectByPrimaryKey1(messageList.get(i).getMessageCommentId()).getCommentInfo());
                    break;
                case Constants.TOPIC_FORBIDEN:
                    map.put("messageId", messageList.get(i).getMessageId());
                    map.put("courseId", messageList.get(i).getMessageCourseId());
                    map.put("courseName", courseDao.selectByPrimaryKey(messageList.get(i).getMessageCourseId()).getCourseName());
                    map.put("forbidenBeginTime", getDate.getYMD(messageList.get(i).getMessageForbidenBeginTime()));
                    map.put("forbidenEndTime", getDate.getYMD(messageList.get(i).getMessageForbidenEndTime()));
                    Topic topic = topicDao.selectByPrimaryKey(messageList.get(i).getMessageTopicId());
                    map.put("topicInfo", topic.getTopicDetail());
                    break;
                case Constants.TRAINING_PRO:
                    map.put("messageId", messageList.get(i).getMessageId());
                    Training training = trainDao.selectByPrimaryKey(messageList.get(i).getMessageExcerciseId());
                    map.put("proExcerciseTime", training.getReleaseDate());
                    map.put("excerciseName", training.getName());
                    break;
                    default:break;
            }
                mapList.add(map);
        }
        map1.put("messageList", mapList);
        int count = 0;
        for (Message message : messageList) {
           if(mMessageStuDao.selectByPrimaryKey(message.getMessageId(),studentId)==null){
                count++;
            }
        }
        map1.put("messageNum", count);
        return map1;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean setMessageState(ArrayList<Integer> messageIdList1,Integer studentId) {
        boolean isInsert=true;
        for(int i=0;i<messageIdList1.size();i++){
            Message_stuKey msgStu=new Message_stuKey();
            msgStu.setMessageId(messageIdList1.get(i));
            msgStu.setStudentId(studentId);
            if(mMessageStuDao.insert(msgStu)!=1){
                isInsert=false;
            }
        }
        return isInsert;
    }

    @Override
    /**
     * 检测指定用户所有作业和测试是否距离截止时间仅剩一天
     */
    public void detectAllHomeWorkEndTime(Integer studentId) {
        //获取登录检测时间
        Date nowDate=new Date();
        ArrayList<Stu_course> stuCourseKeyList=(ArrayList<Stu_course>)stu_course.selectByStudentId(studentId);
        for(int i=0;i<stuCourseKeyList.size();i++){
            //检测该条记录对应的课程期末测试是否发布，如果发布，则检测是否只剩两天，并产生消息提醒
            if(stuCourseKeyList.get(i).getExamScore().equals(Constants.NOT_FINISH_EXAM)){
                Course course=courseDao.selectByPrimaryKey(stuCourseKeyList.get(i).getCourseId());
                //如果已经发布期末测试，则检测是否只剩2天
                if(Constants.EXAM_PASS.equals(course.getExamPass())){
                    Date endDate=course.getExamEndTime();
                    if(this.isLastTwoDay(nowDate,endDate)){
                        System.out.println("CourseServiceImpl.detectAllHomeWorkEndTime:exam just 2 days last");
                        Message msg=new Message();
                        msg.setMessageType(Constants.EXAM_END);
                        msg.setMessageCourseId(stuCourseKeyList.get(i).getCourseId());
                        msg.setMessageExamId(stuCourseKeyList.get(i).getCourseId());
                        msg.setMessageTime(new Date());
                        Message msg1=messageDao.selectByMessageTypeAndExamId(msg.getMessageType(),msg.getMessageExamId());
                        //如果没有这条提醒记录，则进行插入
                        if(msg1==null){
                            messageDao.insertSelective(msg);
                        }
                    }
                }
            }
            ArrayList<Chapter> chapterList=chapterDao.selectByCourseId(stuCourseKeyList.get(i).getCourseId());
            //获取本课程中的所有作业
            for(int j=0;j<chapterList.size();j++){
                ArrayList<Test> testList=testDao.selectByChapterId(chapterList.get(j).getChapterId());
                for(int k=0;k<testList.size();k++){
                    Date endDate=testList.get(k).getTestEndTime();
                    //如果是只剩2天，则产生一条提醒作业截止的message记录
                    if(this.isLastTwoDay(nowDate,endDate)){
                        Message msg=new Message();
                        msg.setMessageType(Constants.HOMEWORK_END);
                        msg.setMessageCourseId(stuCourseKeyList.get(i).getCourseId());
                        msg.setMessageTestId(testList.get(k).getTestId());
                        msg.setMessageTime(new Date());
                        Message msg1=messageDao.selectByMessageTypeAndTestId(msg.getMessageType(),msg.getMessageTestId());
                        //如果没有这条提醒记录，则进行插入
                        if(msg1==null){
                            messageDao.insertSelective(msg);
                        }
                    }
                }

            }
        }
    }

    @Override
    /**
     * 定期清理消息记录
     */
    public void deleteMessage() {
        int betweenDay=7;
        Date dateNow=new Date();
        long Days=dateNow.getTime()/getDate.DAY;
        //每隔七天，处理一次消息记录中超过60天的记录
        if(Days%betweenDay==0){
            System.out.println("CourseServiceImpl.deleteMessage");
            messageDao.deleteByTime();
            }
        }

    @Override
    public HashMap<String, Object> setCheckCourseState(CheckCourse checkCourse) {
        HashMap<String,Object> map=new HashMap<>(1);
        if(mCheckCourseDao.updateByPrimaryKeySelective(checkCourse)==1){
            map.put("result","ok");
        }else {
            map.put("result","error");
        }
        return map;
    }

    @Override
    public JSONArray getAllCourseCheck() {
        List<CheckCourse> ccList=mCheckCourseDao.selectByCheckState();
        JSONArray jsonArr=new JSONArray();
        Gson gs=new Gson();
        for(int i=0;i<ccList.size();i++){
            String jstr=gs.toJson(ccList.get(i));
            JSONObject jsob= new JSONObject();
            try {
                jsob = (JSONObject)new JSONParser().parse(jstr);
            } catch (ParseException e) {
                e.printStackTrace();
            }
            jsonArr.add(jsob);
        }
        return jsonArr;
    }


    @Override
    public JSONObject getCourseAndExam(HttpSession session, Integer courseid) {
        Course course=courseDao.selectByPrimaryKey(courseid);
        Student stu=(Student)session.getAttribute("studentInfo");
        String finishState="";
        Stu_course stuCKey = stu_course.selectByStudentIdAndCourseId(stu.getStudentId(),courseid);
        if(!Constants.NOT_FINISH_EXAM.equals(stuCKey.getExamScore())){
            finishState="finish";
        }else{
            finishState="notFinish";
        }
        StudentExam stuExam=null;
        if(finishState.equals("finish")){
            stuExam=new StudentExam(course,stuCKey);
        }else{
            stuExam=new StudentExam(course);
        }
        stuExam.setIsFinish(finishState);
        if(Constants.EXAM_PASS.equals(stuExam.getExamPass())) {
            if (getDate.DaysBetweenTwoDay(new Date(), course.getExamEndTime()) >= 0) {
                stuExam.setIsEnd("notEnd");
            } else {
                stuExam.setIsEnd("end");
            }
        }else {
            stuExam.setIsEnd("");
        }
        String jstr=new Gson().toJson(stuExam);
        JSONObject jsob=new JSONObject();
        try {
            jsob=(JSONObject)new JSONParser().parse(jstr);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        return jsob;
    }

    @Override
    public JSONObject getCourseAndExam(Integer studentId, Integer courseId) {
        Course course=courseDao.selectByPrimaryKey(courseId);
        StudentExam stuExam=null;
        Stu_course stuCKey = stu_course.selectByStudentIdAndCourseId(studentId,courseId);
        stuExam=new StudentExam(course,stuCKey);
        String jstr=new Gson().toJson(stuExam);
        JSONObject jsob=new JSONObject();
        try {
            jsob=(JSONObject)new JSONParser().parse(jstr);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        return jsob;
    }

    @Override
    public String submitExam(Integer studentId, Integer courseId, String one, String two, String three, String four, String five) {
        Stu_course stuC=new Stu_course();
        stuC.setCourseId(courseId);
        stuC.setOneAnswerSrc(one);
        stuC.setTwoAnswerSrc(two);
        stuC.setThreeAnswerSrc(three);
        stuC.setFourAnswerSrc(four);
        stuC.setFiveAnswerSrc(five);
        stuC.setStudentId(studentId);
        stuC.setExamScore(Constants.FINISH_NOT_READ);
        String result="";
        if(stu_course.updateByPrimaryKeySelective(stuC)==1){
            result="success";
        }else{
            result="fail";
        }
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public String checkExam(Teach_courseKey tc) {
        if(teach_courseDao.updateByPrimaryKeySelective(tc)==1){
            //如果同意，则检查本门课程当中是否全部教师都同意，若是，则置examPass字段为通过
            if(Constants.EXAM_AGREE.equals(tc.getExamAgree())){
                ArrayList<Teach_courseKey> tcList=(ArrayList)teach_courseDao.selectByCourseId(tc.getCourseId());
                boolean agree=true;
                for(int i=0;i<tcList.size();i++){
                    if(!Constants.EXAM_AGREE.equals(tcList.get(i).getExamAgree())){
                        agree=false;
                        break;
                    }
                }
                if(agree){
                    Course course=new Course();
                    course.setExamPass(Constants.EXAM_PASS);
                    course.setCourseId(tc.getCourseId());
                    courseDao.updateByPrimaryKeySelective(course);
                    Message msg=new Message();
                    msg.setMessageType(Constants.COURSE_EXAM_PRO);
                    msg.setMessageCourseId(course.getCourseId());
                    msg.setMessageExamId(course.getCourseId());
                    msg.setMessageTime(new Date());
                    messageDao.insertSelective(msg);
                }
            }
            return "success";
        }
        return "error";
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public String deleteExam(Integer courseId) {
        Course course=new Course();
        course.setCourseId(courseId);
        course.setExamPass(Constants.EXAM_WAIT);
        course.setExamPublishTeacherId(-1);
        course.setExamOneTitle("");
        course.setExamOneAnswer("");
        course.setExamTwoTitle("");
        course.setExamTwoAnswer("");
        course.setExamThreeTitle("");
        course.setExamThreeAnswer("");
        course.setExamFourTitle("");
        course.setExamFourAnswer("");
        course.setExamFiveTitle("");
        course.setExamFiveAnswer("");
        course.setExamSuplement("");
        if(courseDao.updateByPrimaryKeySelective(course)==1){
            ArrayList<Teach_courseKey> tcList=(ArrayList)teach_courseDao.selectByCourseId(course.getCourseId());
            for(int i=0;i<tcList.size();i++){
                tcList.get(i).setExamAgree(Constants.EXAM_AGREE_DEFAULT);
                teach_courseDao.updateByPrimaryKeySelective(tcList.get(i));
            }
            return "success";
        }
        return "error";
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public int submitExamInfo(Course course,HttpSession session) {
        ArrayList<Teach_courseKey> tcList=(ArrayList)teach_courseDao.selectByCourseId(course.getCourseId());
        Teach_courseKey tc=new Teach_courseKey();
        Teacher teac=(Teacher)session.getAttribute("teacherInfo");
        tc.setTeacherId(teac.getTeacherId());
        tc.setCourseId(course.getCourseId());
        tc.setExamAgree(Constants.EXAM_AGREE);
        tc.setExamArgumentSuplement("pass");
        if(teach_courseDao.updateByPrimaryKeySelective(tc)==1){
        }
        //如果教这门课程的老师只有一个，则直接可以发布期末测试，否则需要其他老师同意
        if(tcList.size()==1){
            course.setExamPass(Constants.EXAM_PASS);
            //产生一条测试发布信息
            Message msg=new Message();
            msg.setMessageType(Constants.COURSE_EXAM_PRO);
            msg.setMessageCourseId(course.getCourseId());
            msg.setMessageExamId(course.getCourseId());
            msg.setMessageTime(new Date());
            messageDao.insertSelective(msg);
        }
        return courseDao.updateByPrimaryKeySelective(course);
    }
    public Double getActivity(Integer studentId) {
        HashMap<String, Object> map1 = new HashMap<String, Object>();
        Double activityValue = 0.0;
        //发表的话题数
        int topics = 0;
        //发表的评论数
        int comments = 0;
        //违规的话题数
        int breakTopicRuleNum = 0;
        //违规的评论数
        int breakCommentRuleNum = 0;
        ArrayList<Topic> topicList;
        ArrayList<Comment> comList;
        topicList=topicDao.selectByStudentId(studentId);
        comList=comM.selectByStudentId(studentId);
        topics = topicList.size();
        comments = comList.size();
        breakTopicRuleNum = userDao.selectStudentById(studentId).getBreakTopicRuleNum();
        breakCommentRuleNum = userDao.selectStudentById(studentId).getBreakComRuleNum();
        activityValue = activityValue(topics, comments, breakTopicRuleNum, breakCommentRuleNum);
        return activityValue;
    }
    public static Double activityValue(int topic, int comment, int breakTopic, int breakComm) {
        Double value ;
        Integer topics = topic - breakTopic;
        Integer comments = comment - breakComm;
        System.out.println("topics:" + topics + ";comments:" + comments);
        //换算成topic
        Double level = comments.doubleValue() / 3 + topics.doubleValue();
        int minCom=16;
        if (level <= minCom) {
            value = level * 6;
        } else {
            value = 100.0;
        }
        return value;
    }

    @Override
    public String submitStudentExamScore(Integer[] examScore, Integer studentId, Integer courseId) {
        Double score=0.0;
        for(int i=0;i<examScore.length;i++){
            score+=examScore[i];
        }
        Stu_course sc=new Stu_course();
        sc.setExamScore(score);
        sc.setStudentId(studentId);
        sc.setCourseId(courseId);
        //接下来统计论坛分数，以及作业分数，计算出总成绩
        Double commentScore=(getActivity(studentId));
        Double testScore=0.0;
        ArrayList<Test_score> tsList=test_scoreDao.selectByStudentIdAndCourseId(studentId,courseId);
        Double testAllMarkByOneHundred=0.0;
        for(int i=0;i<tsList.size();i++){
           Test test= testDao.selectByPrimaryKey(tsList.get(i).getTestId());
            Double percent=0.0;
           try {
                percent = tsList.get(i).getScore() / Double.parseDouble(test.getTestAllMark().toString());
           }catch(Exception e){
               System.out.println(e.getMessage());
               percent=0.0;
           }
          testAllMarkByOneHundred+=100*percent;
        }
        int testNum=0;
        ArrayList<Chapter> chapterList=chapterDao.selectByCourseId(courseId);
        for(int i=0;i<chapterList.size();i++){
            testNum+=testDao.selectByChapterId(chapterList.get(i).getChapterId()).size();
        }
        if(testNum==0){
            testScore=100.0;
        }else{
            testScore=testAllMarkByOneHundred/testNum;
        }
        sc.setCommentScore(commentScore);
        sc.setTestScore(testScore);
        Course course=courseDao.selectByPrimaryKey(courseId);
        Double commentWeight=course.getEvaluationChatWeight();
        Double testWeight=course.getEvaluationTestWeight();
        Double examWeight=course.getEvaluationExamWeight();
        Double allScore=commentScore*commentWeight+testWeight*testScore+examWeight*score;
        sc.setAllScore(allScore);
        if(stu_course.updateByPrimaryKeySelective(sc)==1){
            return "success";
        }
        return "error";
    }

    @Override
    public JSONArray getFinalExamScore(Integer studentId) {
        ArrayList<Stu_course> stucList=(ArrayList<Stu_course>)stu_course.selectByStudentId(studentId);
        ArrayList<Object> objectList=new ArrayList<>();
        for(int i=0;i<stucList.size();i++){
            if(!stucList.get(i).getExamScore().equals(Constants.NOT_FINISH_EXAM)&&!stucList.get(i).getExamScore().equals(Constants.FINISH_NOT_READ)){
                StuCourse stuc=new StuCourse(stucList.get(i));
                stuc.setCourseName(courseDao.selectByPrimaryKey(stuc.getCourseId()).getCourseName());
                objectList.add(stuc);
            }
        }
        return GsonUtils.objectListToJSONArray(objectList);
    }

    /**
     *
     * @param nowDate
     * @param endDate
     * @return boolean
     */
    private boolean isLastTwoDay(Date nowDate, Date endDate) {
        int large=2;
        int small=1;
        long betweenDay=getDate.DaysBetweenTwoDay(nowDate,endDate);
        if(betweenDay<=large&&betweenDay>=small){
            return true;
        }
        return false;
    }


    public ArrayList<HashMap<String, Object>> filterMapList(ArrayList<HashMap<String, Object>> mapList) {
        for (int i = 0; i < mapList.size(); i++) {
            if (((ArrayList<HashMap<String, Object>>) mapList.get(i).get("testOrExamList")).size() == 0&&((ArrayList<HashMap<String, Object>>) mapList.get(i).get("examList")).size()==0) {
                mapList.remove(i);
            }
        }
        return mapList;
    }

    public Double getPassPerc(ArrayList<Test_score> tsList) {
        int count = 0;
        for (int i = 0; i < tsList.size(); i++) {
            if (tsList.get(i).getScore() >= 60) {
                count++;
            }
        }
        if (tsList.size() != 0) {
            return (count / ((double) tsList.size())) * 100;
        }
        return 0.0;
    }

    public Double getGoodPerc(ArrayList<Test_score> tsList) {
        int count = 0;
        for (int i = 0; i < tsList.size(); i++) {
            if (tsList.get(i).getScore() >= 80) {
                count++;
            }
        }
        if (tsList.size() != 0) {
            return (count / ((double) tsList.size())) * 100;
        }
        return 0.0;
    }

    public List<HashMap<String, Object>> toMapList(ArrayList<CheckCourse> ccList) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        for (int i = 0; i < ccList.size(); i++) {
            HashMap<String, Object> map1 = new HashMap<>();
            map1.put("courseName", ccList.get(i).getCourseName());
            map1.put("checkId", ccList.get(i).getCheckId());
            if (Constants.CHECK_COURSE_WAIT_CHECK.equals(ccList.get(i).getCheckState())) {
                map1.put("checkState", "待审核");
            } else if (Constants.CHECK_COURSE_PASS.equals(ccList.get(i).getCheckState())) {
                map1.put("checkState", "申请成功");
                map1.put("suplement", ccList.get(i).getSuplement());
            } else {
                map1.put("checkState", "申请失败");
                map1.put("suplement", ccList.get(i).getSuplement());
            }
            mapList.add(map1);
        }
        return mapList;
    }

    public ArrayList<HashMap<String, Object>> getTestDetail(Integer testId) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        //获取指定TestId的大题集合
        ArrayList<TestProblem> testProblemList = testProblemDao.selectByTestId(testId);
        for (int i = 0; i < testProblemList.size(); i++) {
            HashMap<String, Object> map1 = new HashMap<>(10);
            //放入类型
            switch (testProblemList.get(i).getTestProblemType())
            {
                case 0:
                    map1.put("testBigType", "choose");
                    break;
                case 1:
                    map1.put("testBigType", "judge");
                    break;
                case 2:
                    map1.put("testBigType", "fillIn");
                    break;
                    default:break;
            }
            ArrayList<HashMap<String, Object>> mapList1 = this.getSmallDetail(testProblemList.get(i).getTestProblemId());
            map1.put("testBigNumber", testProblemList.get(i).getTestProblemOrder());
            //小题细节
            map1.put("smallDetail", mapList1);
            map1.put("smallNum", mapList1.size());
            map1.put("smallScore", mapList1.get(0).get("smallScore"));
            map1.put("allScore", this.getAllScore(mapList1));
            mapList.add(map1);
        }
        return mapList;
    }

    public int getAllScore(ArrayList<HashMap<String, Object>> smallProblemList) {
        int score = 0;
        for (int i = 0; i < smallProblemList.size(); i++) {
            score += Integer.parseInt(smallProblemList.get(i).get("smallScore").toString());
        }
        return score;
    }

    public ArrayList<HashMap<String, Object>> getSmallDetail(Integer TestProblemId) {
        //一道大题包含若干个小题
        ArrayList<smallProblem> smallProblemList = smallProblemDao.selectByTestProblemId(TestProblemId);
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        for (int i = 0; i < smallProblemList.size(); i++) {
            HashMap<String, Object> map1 = new HashMap<>(15);
            map1.put("title", smallProblemList.get(i).getSmallProblemTitle());
            map1.put("smallScore", smallProblemList.get(i).getSmallProblemScore());
            map1.put("trueAnswer", smallProblemList.get(i).getSmallProblemTrueAnswer());
            map1.put("answer", smallProblemList.get(i).getSmallProblemTip());
            map1.put("A", smallProblemList.get(i).getSmallProblemADetail());
            map1.put("B", smallProblemList.get(i).getSmallProblemBDetail());
            map1.put("C", smallProblemList.get(i).getSmallProblemCDetail());
            map1.put("D", smallProblemList.get(i).getSmallProblemDDetail());
            mapList.add(map1);
        }
        //返回指定大题ID的所有小题信息
        return mapList;
    }

    public static ArrayList<String> segment(String str, char ch) {
        ArrayList<String> li = new ArrayList<>();
        int start = 0;
        int end ;
        int i ;
        boolean notExistCh=true;
        for (i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ch) {
                notExistCh=false;
                String sb ;
                end = i;
                sb = str.substring(start, end);
                li.add(sb);
                start = i + 1;
            }
        }
        //默认最后没有分隔符,例如"语文|数学|英语"
        if(!notExistCh) {
            end = i;
            li.add(str.substring(start, end));
        }
        else{
            li.add(str);
        }
        return li;
    }

    public String getBiggest(List<String> hh) {
        int i = 0;
        int j = 0;
        ArrayList<ArrayList<String>> lists = new ArrayList<>();
        for (i = 0; i < hh.size(); i++) {
            boolean fl = false;
            if (lists.size() != 0) {
                for (j = 0; j < lists.size(); j++) {
                    if (lists.get(j).get(0).equals(hh.get(i))) {
                        lists.get(j).add(hh.get(i));
                        fl = true;
                        break;
                    }
                }
            }
            if (fl == false) {
                ArrayList<String> shad = new ArrayList<>();
                shad.add(hh.get(i));
                lists.add(shad);
            }
        }
        String big = "";
        int co = 0;
        for (int k = 0; k < lists.size(); k++) {
            if (lists.get(k).size() > co) {
                big = lists.get(k).get(0);
                co = lists.get(k).size();
            }
        }
        return big;
    }


    @Override
    public ArrayList<HashMap<String, Object>> toHashMap(List<Course> courseList, int num) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<>();
        for (int i = 0; i < num; i++) {
            HashMap<String, Object> map1 = new HashMap<>();
            map1.put("courseId", courseList.get(i).getCourseId());
            map1.put("courseImgSrc", courseList.get(i).getCoursePostSrc());
            map1.put("courseName", courseList.get(i).getCourseName());
            SchOrCom sch1 = schoolDao.selectByPrimaryKey(courseList.get(i).getBelongSchId());
            map1.put("school", sch1.getComName());
            ArrayList<Teach_courseKey> teacherCourseKeyList=(ArrayList<Teach_courseKey>)teach_courseDao.selectByCourseId(courseList.get(i).getCourseId());
            String str="";
            for(int j=0;j<teacherCourseKeyList.size();j++){
                str+=" "+teacherDao.selectByPrimaryKey(teacherCourseKeyList.get(j).getTeacherId()).getTeacherName();
            }
            map1.put("teachers", str);
            List<Stu_course> stu_c_List = stu_course.selectByCourseId(courseList.get(i).getCourseId());
            map1.put("fans", stu_c_List.size());
            map1.put("evaluationinfo", courseList.get(i).getEvaluationLevel());
            map1.put("comments", commentDao.selectByCourseId(courseList.get(i).getCourseId()).size());
            map1.put("courseIntroduce", courseList.get(i).getCourseIntroduce());
            map1.put("Progress", courseList.get(i).getCourseProgress());
            int chapterNum = chapterDao.selectByCourseId(courseList.get(i).getCourseId()).size();
            map1.put("chapterNum", chapterNum);
            map1.put("evaluationTestWeight", courseList.get(i).getEvaluationTestWeight());
            map1.put("evaluationExamWeight", courseList.get(i).getEvaluationExamWeight());
            map1.put("evaluationChatWeight", courseList.get(i).getEvaluationChatWeight());
            map1.put("courseType",courseList.get(i).getCourseType());
            map1.put("comPic", sch1.getComPic());
            int progress = courseList.get(i).getCourseProgress();
            if (progress == 0) {
                map1.put("courseState", "未开课");
            } else if (progress < chapterNum) {
                map1.put("courseState", "正在进行");
            } else {
                map1.put("courseState", "已完结");
            }
            mapList.add(map1);
        }
        return mapList;
    }

    public HashMap<String, Object> toMap(Course course,HttpSession session) {
        HashMap<String, Object> map1 = new HashMap<String, Object>();
        Student student = (Student) session.getAttribute("studentInfo");
        Teacher teacher = (Teacher) session.getAttribute("teacherInfo");
        if (student != null) {
            if (stu_course.selectByStudentIdAndCourseId(student.getStudentId(), course.getCourseId()) != null) {
                map1.put("isChoose", "true");
            } else {
                map1.put("isChoose", "false");
            }
            map1.put("isManage", "false");
        } else if (teacher != null) {
            if (teach_courseDao.selectByTeacherIdAndCourseId(teacher.getTeacherId(), course.getCourseId()) != null) {
                map1.put("isManage", "true");
                ArrayList<HashMap<String,Object>> mapList=new ArrayList<>();
                ArrayList<Teach_courseKey> tcList=(ArrayList<Teach_courseKey>)teach_courseDao.selectByCourseId(course.getCourseId());
                for(int i=0;i<tcList.size();i++){
                    if(tcList.get(i).getTeacherId().equals(teacher.getTeacherId())){
                        continue;
                    }else{
                        HashMap<String,Object> map2=new HashMap<>(4);
                        map2.put("checkTeacher",teacherDao.selectByPrimaryKey(tcList.get(i).getTeacherId()).getTeacherName());
                        if(tcList.get(i).getExamAgree().equals(Constants.EXAM_AGREE)){
                            map2.put("checkResult","通过");
                        }else if(tcList.get(i).getExamAgree().equals(Constants.EXAM_WAIT)){
                            map2.put("checkResult","待确认");
                        }else{
                            map2.put("checkResult","不通过");
                        }
                        map2.put("checkSuplement",tcList.get(i).getExamArgumentSuplement());
                        mapList.add(map2);
                    }
                }
                map1.put("checkInfo",mapList);
                ArrayList<Stu_course> stuCList=(ArrayList<Stu_course>)stu_course.selectByCourseId(course.getCourseId());
                ArrayList<HashMap<String,Object>> mapList1=new ArrayList<>();
                for(int i=0;i<stuCList.size();i++){
                    //如果完成了，但未批阅，则加入之
                    if(stuCList.get(i).getExamScore().equals(Constants.FINISH_NOT_READ)){
                        HashMap<String,Object> map2=new HashMap<>(3);
                        map2.put("studentId",stuCList.get(i).getStudentId());
                        Student stud=studentDao.selectByPrimaryKey(stuCList.get(i).getStudentId());
                        map2.put("studentAccount",stud.getStudentAccount());
                        map2.put("studentName",stud.getStudentName());
                        mapList1.add(map2);
                    }
                }
                map1.put("studentExamList",mapList1);
            } else {
                map1.put("isManage", "false");
            }
            map1.put("isChoose", "false");
        } else {
            map1.put("isManage", "false");
            map1.put("isChoose", "false");
        }
        map1.put("courseId", course.getCourseId());
        map1.put("courseImgSrc", course.getCoursePostSrc());
        map1.put("courseName", course.getCourseName());
        SchOrCom sch1 = schoolDao.selectByPrimaryKey(course.getBelongSchId());
        map1.put("school", sch1.getComName());
        ArrayList<Teach_courseKey> teacherCourseKeyList=(ArrayList<Teach_courseKey>)teach_courseDao.selectByCourseId(course.getCourseId());
        String str="";
        for(int j=0;j<teacherCourseKeyList.size();j++){
            str+=" "+teacherDao.selectByPrimaryKey(teacherCourseKeyList.get(j).getTeacherId()).getTeacherName();
        }
        map1.put("teachers", str);
        List<Stu_course> stu_c_List = stu_course.selectByCourseId(course.getCourseId());
        map1.put("fans", stu_c_List.size());
        map1.put("comments", commentDao.selectByCourseId(course.getCourseId()).size());
        map1.put("courseIntroduce", course.getCourseIntroduce());
        map1.put("Progress", course.getCourseProgress());
        int chapterNum = chapterDao.selectByCourseId(course.getCourseId()).size();
        map1.put("chapterNum", chapterNum);
        map1.put("evaluationinfo", course.getEvaluationLevel());
        map1.put("evaluationTestWeight", course.getEvaluationTestWeight());
        map1.put("evaluationExamWeight", course.getEvaluationExamWeight());
        map1.put("evaluationChatWeight", course.getEvaluationChatWeight());
        map1.put("comPic", sch1.getComPic());
        map1.put("introduceVideoSrc", course.getIntroduceVideoSrc());
        int progress = course.getCourseProgress();
        map1.put("examEndTime",getDate.getYMD(course.getExamEndTime()));
        map1.put("examStartTime",getDate.getYMD(course.getExamStartTime()));
        map1.put("examFiveAnswer",course.getExamFiveAnswer());
        map1.put("examFiveTitle",course.getExamFiveTitle());
        map1.put("examFourAnswer",course.getExamFourAnswer());
        map1.put("examFourTitle",course.getExamFourTitle());
        map1.put("examThreeAnswer",course.getExamThreeAnswer());
        map1.put("examThreeTitle",course.getExamThreeTitle());
        map1.put("examTwoAnswer",course.getExamTwoAnswer());
        map1.put("examTwoTitle",course.getExamTwoTitle());
        map1.put("examOneAnswer",course.getExamOneAnswer());
        map1.put("examOneTitle",course.getExamOneTitle());
        map1.put("examLimitTime",course.getExamLimitTime());
        map1.put("examPublishTeacherId",course.getExamPublishTeacherId());
        map1.put("examSuplement",course.getExamSuplement());
        map1.put("examPass",course.getExamPass());
        if (progress == 0) {
            map1.put("courseState", "未开课");
        } else if (progress < chapterNum) {
            map1.put("courseState", "正在进行");
        } else {
            map1.put("courseState", "已完结");
        }
        return map1;
    }

    public List<HashMap<String, Object>> toHashMap1(ArrayList<Teacher> teacherList, int num) {
        ArrayList<HashMap<String, Object>> mapList = new ArrayList<HashMap<String, Object>>();
        for (int i = 0; i < num; i++) {
            HashMap<String, Object> map1 = new HashMap<String, Object>();
            map1.put("teacherName", teacherList.get(i).getTeacherName());
            map1.put("commentNum", teacherList.get(i).getCommentNum());
            map1.put("schoolId", teacherList.get(i).getBelongSchId());
            map1.put("createDate", teacherList.get(i).getCreateDate());
            map1.put("schoolName", teacherList.get(i).getBelongSchName());
            map1.put("teacherId", teacherList.get(i).getTeacherId());
            map1.put("teacherAccount", teacherList.get(i).getTeacherAccount());
            map1.put("teacherPassword", teacherList.get(i).getTeacherPassword());
            map1.put("teacherIntroduce", teacherList.get(i).getTeacherIntroduce());
            map1.put("teacherPic", teacherList.get(i).getTeacherPic());
            map1.put("teacherLevel", teacherList.get(i).getTeacherLevel());
            map1.put("teacherSex", teacherList.get(i).getTeacherSex());
            mapList.add(map1);
        }
        return mapList;
    }

    public void sort(ArrayList<HashMap<String, Object>> mapList2, int low, int high) {
        int pivot;
        if (low < high) {
            pivot = findPivot(mapList2, low, high);
            //左边序列。第一个索引位置到关键值索引-1
            sort(mapList2, low, pivot - 1);
            //右边序列。从关键值索引+1到最后一个
            sort(mapList2, pivot + 1, high);
        }
    }

    public int findPivot(ArrayList<HashMap<String, Object>> mapList2, int low, int high) {
        HashMap<String, Object> map1 = new HashMap<String, Object>();
        map1.putAll(mapList2.get(low));
        //得到排序的关键字
        int key = Integer.parseInt(mapList2.get(low).get("fans").toString());
        while (low < high) {
            //如果没有比关键值小的，比较下一个，直到有比关键值小的交换位置，然后又从前往后比较
            while (low < high && Integer.parseInt(mapList2.get(high).get("fans").toString()) <= key) {
                high--;
            }
            mapList2.get(low).putAll(mapList2.get(high));
            //如果没有比关键值大的，比较下一个，直到有比关键值大的交换位置
            while (low < high && Integer.parseInt(mapList2.get(low).get("fans").toString()) >= key) {
                low++;
            }
            mapList2.get(high).putAll(mapList2.get(low));
        }
        mapList2.get(low).putAll(map1);
        return low;
    }

    public void swap(HashMap<String, Object> map1, HashMap<String, Object> map2) {
        HashMap<String, Object> temp = new HashMap<>();
        temp.putAll(map1);
        map1.putAll(map2);
        map2.putAll(temp);
    }
}
