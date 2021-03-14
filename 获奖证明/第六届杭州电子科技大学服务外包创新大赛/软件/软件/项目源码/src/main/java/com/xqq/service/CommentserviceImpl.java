package com.xqq.service;

import com.xqq.constant.Constants;
import com.xqq.dao.*;
import com.xqq.pojo.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.annotation.Resource;
import javax.servlet.http.HttpSession;
import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

@Service("commentService")
public class CommentserviceImpl implements CommentService {
    @Resource(name = "CommentMapper")
    CommentMapper comM;
    @Resource(name = "Praise_commentMapper")
    Praise_commentMapper praiseCOM;
    @Resource(name = "UserDao")
    IUserDao userDao;
    @Resource(name = "TeacherMapper")
    TeacherMapper teacherDao;
    @Resource
    TopicMapper topicDao;
    @Resource
    MessageMapper messageDao;
    @Resource
    StudentMapper studentDao;

    /**
     *
     * @param com
     * @param userType
     * @param userId
     * @return
     * userType 和 userId 是用来判断当前登录用户是否对本条评论点赞
     */
    public HashMap<String, Object> toMap(Comment com, short userType, int userId) {
        String writterName, writterTime, writterInfo;//接口参数
        int PraiseNum, ReplyNum, writterId, writterType, replyToId, commentId, reportNum;
        HashMap<String, Object> map = new HashMap<>(15);
        replyToId = com.getReplyToId();
        //评论者ID
        writterId = com.getCommentWritterId();
        //评论时间
        writterTime = com.getCommentTime();
        //评论内容
        writterInfo = com.getCommentInfo();
        //评论者身份
        writterType = com.getCommentWritterType();
        //评论类型(回复/发表，若值为-1则是发表)
        commentId = com.getCommentId();
        reportNum = com.getReportNum();
        //如果是学生
        if (com.getCommentWritterType() == 0) {
            Student stu1 = userDao.selectStudentById(writterId);
            //获取名字
            writterName = stu1.getStudentName();
        } else {
            Teacher teacher = teacherDao.selectByPrimaryKey(writterId);
            //获取名字
            writterName = teacher.getTeacherName();
        }

        Praise_commentKey pc1 = new Praise_commentKey();
        pc1.setCommentId(commentId);
        pc1.setPraiseUserId(userId);
        pc1.setPraiseUserType(userType);
        Praise_commentKey pc2 = praiseCOM.selectByPrimaryKey(pc1);
        if (pc2 != null) {
            map.put("isAlreadyPraise", "orange");
        } else {
            map.put("isAlreadyPraise", "grey");
        }
        List<Praise_commentKey> p_cmList = praiseCOM.selectByCommentId(commentId);
        //获取点赞量
        PraiseNum = p_cmList.size();
        List<Comment> replyList = comM.selectReply(commentId);
        //获取回复数
        ReplyNum = replyList.size();
        map.put("writterId", writterId);
        map.put("writterName", writterName);
        map.put("writterTime", writterTime);
        map.put("reportNum", reportNum);
        if (writterType == 1) {
            map.put("writterTypeInfo", "老师");
        } else {
            map.put("writterTypeInfo", "学生");
        }
        map.put("writterType", writterType);
        //学生
        if (writterType == 0) {
            Student stu = userDao.selectStudentById(writterId);
            map.put("writterImg", stu.getStudentPic());
        }
        //老师
        else if (writterType == 1) {
            Teacher teac = teacherDao.selectByPrimaryKey(writterId);
            map.put("writterImg", teac.getTeacherPic());
        }
        map.put("PraiseNum", PraiseNum);
        map.put("ReplyNum", ReplyNum);
        map.put("ReplyToId", replyToId);
        map.put("writterInfo", writterInfo);
        map.put("commentId", commentId);
        //如果是回复类型
        if (replyToId != -1) {
            //找到回复对象的评论信息
            Comment comment = comM.selectByPrimaryKey1(replyToId);
            if(Constants.IS_FORBID.equals(comment.getIsForbiden())){
                String forbidSentence="该评论已被禁止，无法查看";
                comment.setCommentInfo(forbidSentence);
            }
            map.put("replyToInfo", comment.getCommentInfo());
            //学生
            if (comment.getCommentWritterType() == 0) {
                Student stu = userDao.selectStudentById(comment.getCommentWritterId());
                map.put("replyToName", stu.getStudentName());
                //老师
            } else if (comment.getCommentWritterType() == 1) {
                Teacher teac = teacherDao.selectByPrimaryKey(comment.getCommentWritterId());
                map.put("replyToName", teac.getTeacherName());
            }
        }
        return map;
    }

    private ArrayList<HashMap<String,Object>> toMapList(ArrayList<Comment> comList,HttpSession session){
        ArrayList<HashMap<String, Object>> comInfo = new ArrayList<>();
        Teacher teacher1 = (Teacher) session.getAttribute("teacherInfo");
        Student student1 = (Student) session.getAttribute("studentInfo");
        //userId 和 userType 是用来判断当前登录用户是否对本条评论点过赞
        Integer userId = -1;
        short userType = -1;
        //当前登录用户是学生
        if (student1 != null)
        {
            userId = student1.getStudentId();
            userType = Constants.TYPE_OF_STUDENT;

        }
        //当前登录用户是老师
        else if (teacher1 != null) {
            userId = teacher1.getTeacherId();
            userType = Constants.TYPE_OF_TEACHER;
        } else {
            System.out.println("showcomment:system login error");
        }
        for (int i = 0; i < comList.size(); i++) {
            //当评论类型为发表时，检测是否被禁止显示，禁止显示则不添加
            if(Constants.NOT_REPLY.equals(comList.get(i).getReplyToId())){
                if(!Constants.IS_FORBID.equals(comList.get(i).getIsForbiden())){
                    HashMap<String,Object> map=this.toMap(comList.get(i),userType,userId);
                    comInfo.add(map);
                }
            }
            //如果评论类型为回复，则检测其回复对象是否被禁止显示，若是，则用"该评论已被禁止，无法查看"代替
            else if(!Constants.IS_FORBID.equals(comList.get(i).getIsForbiden())){
                HashMap<String,Object> map=this.toMap(comList.get(i),userType,userId);
                comInfo.add(map);
            }
        }
        return comInfo;
    }
    @Override
    public ArrayList<HashMap<String, Object>> showComment(Integer topicId,HttpSession session) {
        //调用dao层实现课程评论区显示功能
        ArrayList<Comment> comList = (ArrayList<Comment>)(comM.selectByTopicId(topicId));
        return this.toMapList(comList,session);
    }
    @Override
    public HashMap<String, Object> writeComment(Comment comm,HttpSession session) {
        HashMap<String, Object> map = new HashMap<>(15);
        Teacher teacher1 = (Teacher) session.getAttribute("teacherInfo");
        Student student1 = (Student) session.getAttribute("studentInfo");
        //如果是学生，先检查该用户是否已被限制发言权
        if(student1!=null){
            ArrayList<Message> messageList=messageDao.selectByMessageTypeAndStudentId(student1.getStudentId());
            ArrayList<Long> timeList=new ArrayList<>() ;
            //得到所有关于该学生评论禁言列表
            for(int i=0;i<messageList.size();i++){
                timeList.add(messageList.get(i).getMessageForbidenEndTime().getTime());
            }
            //如果有禁言的话
            if(timeList.size()!=0){
                Long largestTime=this.getBiggestNum(timeList);
                Date dateNow=new Date();
                //如果禁言时间还未结束
                if(largestTime>dateNow.getTime()){
                    map.put("result","forbiden");
                    return map;
                }
            }

        }
        if (comM.insertSelective(comm) == 1) {
            Comment comment1 = comM.selectByTimeAndWritter(comm.getCommentWritterId(), comm.getCommentWritterType(), comm.getCommentTime());
            Integer userId = -1;
            short userType = -1;
            //当前登录用户是学生
            if (student1 != null)
            {
                userId = student1.getStudentId();
                userType = Constants.TYPE_OF_STUDENT;
            }
            //当前登录用户是老师
            else if (teacher1 != null) {
                System.out.print("");
                userId = teacher1.getTeacherId();
                userType = Constants.TYPE_OF_TEACHER;
            } else {
                System.out.println("showcomment:system login error");
            }
            map = toMap(comment1,userType,userId);
            map.put("result","ok");
            Message msg=new Message();
            msg.setMessageTime(new Date());
            Integer topicId=comM.selectByPrimaryKey(comment1.getCommentId()).getBelongTopicId();
            msg.setMessageTopicId(topicId);
            msg.setMessageCourseId(topicDao.selectByPrimaryKey(topicId).getBelongCourseId());
            msg.setMessageCommentId(comment1.getCommentId());
            if(comment1.getReplyToId().equals(Constants.NOT_REPLY)){
                msg.setMessageType(com.xqq.constant.Constants.TOPIC_NEW_COMMENTS);
            }
            else{
                msg.setMessageType(Constants.COMMENT_REPLY);
            }
            messageDao.insertSelective(msg);
        }
        return map;
    }
    public long getBiggestNum(ArrayList<Long> timeList){
        long largest=0;
        for(int i=0;i<timeList.size();i++){
            if(timeList.get(i)>largest){
                largest=timeList.get(i);
            }
        }
        return largest;
    }
    @Override
    public int praise(Praise_commentKey pra) {
        //点过赞再点一次就取消
        if (praiseCOM.selectByPrimaryKey(pra) != null)
        {
            if (praiseCOM.deleteByPrimaryKey(pra) == 1) {
                return 1;
            }
        } else {
            if (praiseCOM.insert(pra) == 1) {

                return 0;
            }
        }
        return -1;
    }

    @Override
    public ArrayList<HashMap<String, Object>> showTopic(Integer courseId,String textPath) {
        ArrayList<HashMap<String, Object>> topicInfo;
        List<Topic> topicList = topicDao.selectByCourseId(courseId);
        topicInfo = this.toMapList(topicList,textPath);
        return topicInfo;
    }

    @Override
    public synchronized boolean writeTopic(Topic topic,String textPath) {
        String path=textPath+"\\"+"Topic_Read_Count.txt";
        Integer topicId = this.getAutoIncreaseTopicId(path);
        String newLineTxt = topicId + "#" + "0";
        if (topicId != -1 && this.writeFile(path, newLineTxt)) {
            if (topicDao.insertSelective(topic) == 1) {
                return true;
            }
        }
        return false;
    }

    public Integer getAutoIncreaseTopicId(String path1) {
        try {
            String encoding = "GBK";
            File file = new File(path1);
            //判断文件是否存在
            if (file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(
                        //考虑到编码格式
                        new FileInputStream(file), encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                StringBuffer buf = new StringBuffer();
                String lineTxt = null;
                ArrayList<String> lineList = new ArrayList<String>();
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    lineList.add(lineTxt);
                }
                ArrayList<String> strArry = CourseServiceImpl.segment(lineList.get(lineList.size() - 1), '#');
                //新加的topicId
                Integer topicId = Integer.parseInt(strArry.get(0)) + 1;
                read.close();
                return topicId;
            } else {
                System.out.println("找不到指定的文件");
                return -1;
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
            return -1;
        }
    }

    public ArrayList<HashMap<String, Object>> showReadCount( String textPath) {//返回浏览信息集合
        try {
            String encoding = "GBK";
            String path=textPath+"\\"+"Topic_Read_Count.txt";
            File file = new File(path);
            //判断文件是否存在
            if (file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(
                        //考虑到编码格式
                        new FileInputStream(file), encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                StringBuffer buf = new StringBuffer();
                String lineTxt = null;
                ArrayList<HashMap<String, Object>> readCList = new ArrayList<HashMap<String, Object>>();
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    ArrayList<String> strArry = CourseServiceImpl.segment(lineTxt, '#');
                    HashMap<String, Object> temp = new HashMap<String, Object>();
                    temp.put("topicId", strArry.get(0));
                    Integer readCounts = Integer.parseInt(strArry.get(1));
                    temp.put("readCount", readCounts);
                    readCList.add(temp);
                }
                read.close();
                return readCList;
            } else {
                System.out.println("找不到指定的文件");
                return null;
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
            return null;
        }

    }

    @Override
    public HashMap<String, Object> readCount(Integer topicId,String dir) {
        HashMap<String, Object> map = new HashMap<>();
        createFile(dir, "temp.txt");
        String toPath = dir+"\\"+"temp.txt";
        String path=dir+"\\"+"Topic_Read_Count.txt";
        readAndCopyByFileLine(path, toPath, topicId,dir);
        map.put("result", "ok");
        return map;
    }

    @Override
    public HashMap<String, Object> reportTopic(Integer topicId) {
        HashMap<String, Object> map = new HashMap<>(1);
        if (topicDao.updateReportNum(topicId) == 1) {
            map.put("result", "ok");
            System.out.println("updateReportNum(topicId)=1");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @Override
    public HashMap<String, Object> reportComment(Integer commentId) {
        HashMap<String, Object> map = new HashMap<String, Object>();
        if (comM.updateReportNum(commentId) == 1) {
            map.put("result", "ok");
            System.out.println("updateReportNum(commentId)=1");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    @Override
    public HashMap<String, Object> getActivity(Integer studentId) {
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
        map1.put("activityValue", activityValue);
        return map1;
    }

    @Override
    public ArrayList<HashMap<String, Object>> showCheckComment() {
        ArrayList<Comment> comList= comM.selectByReportNum(Constants.MAX_REPORT_NUM);
        ArrayList<HashMap<String,Object>> mapList=new ArrayList<>();
        for(int i=0;i<comList.size();i++){
            HashMap<String,Object> map=new HashMap<>(4);
            map.put("commentWriterId",comList.get(i).getCommentWritterId());
            map.put("commentWriterName",userDao.selectStudentById(comList.get(i).getCommentWritterId()).getStudentName());
            map.put("commentTime",comList.get(i).getCommentTime());
            map.put("commentInfo",comList.get(i).getCommentInfo());
            map.put("commentId",comList.get(i).getCommentId());
            mapList.add(map);
        }
        return mapList;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public HashMap<String, Object> forbidenComment(Message msg) {
        HashMap<String,Object> map=new HashMap<>(1);
        Integer belongTopicId=comM.selectByPrimaryKey(msg.getMessageCommentId()).getBelongTopicId();
        Integer belongCourseId=topicDao.selectByPrimaryKey(belongTopicId).getBelongCourseId();
        msg.setMessageCourseId(belongCourseId);
        msg.setMessageType(Constants.COMMENT_FORBIDEN);
            comM.updateByCommentId(msg.getMessageCommentId());
            studentDao.updateBreakComRule(msg.getMessageStudentId());
        messageDao.insertSelective(msg);
        map.put("result","已将该用户禁言");
        return map;
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

    public HashMap<String, Object> deleteTopic(Integer topicId,String dir) {
        HashMap<String, Object> map = new HashMap<String, Object>();
        createFile(dir, "temp1.txt");
        String toPath = dir+"\\temp1.txt";
        String path=dir+"\\"+"Topic_Read_Count.txt";
        //第一步先删除topic的浏览记录
        deleteReadCountDocByFileLine(path, toPath, topicId,dir);
        //第二步，进入数据库删除对应的topic记录
        if (this.deleteTopicInDB(topicId)) {
            map.put("result", "ok");
        } else {
            map.put("result", "no");
        }
        return map;
    }

    public boolean deleteTopicInDB(Integer topicId) {//在数据库中删除topic
        if (topicDao.deleteByPrimaryKey(topicId) == 1)
            return true;
        return false;
    }

    public static String deleteReadCountDocByFileLine(String path, String toPath, int topicId,String TextPath) {
        try {
            String encoding = "GBK";
            File file = new File(path);
            //判断文件是否存在
            if (file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(
                        //考虑到编码格式
                        new FileInputStream(file), encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                StringBuffer buf = new StringBuffer();
                String lineTxt = null;
                //统计行数
                int count = 0;
                PrintWriter pw = new PrintWriter(new FileWriter(toPath, false));
                while (count < topicId - 1 && (lineTxt = bufferedReader.readLine()) != null
                        && Integer.parseInt(CourseServiceImpl.segment(lineTxt, '#').get(0))
                        != topicId)//当本行topicId不等于指定topicId 时执行拷贝
                {
                    count++;
                    pw.println(lineTxt);

                }
                // 中间自然就跳过了需要删除的那行记录
                //继续拷贝剩下的数据
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    pw.println(lineTxt);
                }
                pw.flush();
                read.close();
                pw.close();
                delete(path);
                //给拷贝文件重命名
                renameFile(TextPath, "temp1.txt", "topic_Read_Count.txt");
                return "true";
            } else {
                System.out.println("找不到指定的文件");
                return "false";
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
            return "false";
        }
    }

    public static String readAndCopyByFileLine(String path, String toPath, int topicId,String TextPath) {
        try {
            String encoding = "GBK";
            File file = new File(path);
            //判断文件是否存在
            if (file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(
                        //考虑到编码格式
                        new FileInputStream(file), encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                StringBuffer buf = new StringBuffer();
                String lineTxt = null;
                PrintWriter pw = new PrintWriter(new FileWriter(toPath, false));
                boolean isCome = false;
                //当本行topicId不等于指定topicId 时执行拷贝
                while ((lineTxt = bufferedReader.readLine()) != null)
                {
                    if (Integer.parseInt(CourseServiceImpl.segment(lineTxt, '#').get(0)) != topicId) {
                        pw.println(lineTxt);
                    } else {
                        isCome = true;
                        break;
                    }

                }
                if (isCome) {
                    //输出topicId这一行的数据
                    System.out.println("needUpdateLine:" + lineTxt);
                }
                ArrayList<String> strArry = CourseServiceImpl.segment(lineTxt, '#');
                int readCounts = Integer.parseInt(strArry.get(1));
                //浏览次数+1
                readCounts += 1;
                String newStr ;
                newStr = strArry.get(0) + "#" + readCounts;
                pw.println(newStr);
                //继续拷贝剩下的数据
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    pw.println(lineTxt);
                }
                pw.flush();
                read.close();
                pw.close();
                delete(path);
                //给拷贝文件重命名
                renameFile(TextPath, "temp.txt", "topic_Read_Count.txt");
                return "true";
            } else {
                System.out.println("找不到指定的文件");
                return "false";
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
            return "false";
        }
    }

    public boolean writeFile(String toPath, String info) {
        try {
            PrintWriter pw = new PrintWriter(new FileWriter(toPath, true));
            pw.println(info);
            pw.flush();
            pw.close();
            return true;
        } catch (IOException e) {
            System.out.println("commentserviceImpl.writeFile:" + e.getMessage());
            return false;
        }
    }

    public static boolean delete(String fileName) {
        File file = new File(fileName);
        if (!file.exists()) {
            System.out.println("删除文件失败:" + fileName + "不存在！");
            return false;
        } else {
            if (file.isFile()) {
                return deleteFile(fileName);
            } else {
                return false;
            }
        }
    }

    public static boolean deleteFile(String fileName) {
        File file = new File(fileName);
        // 如果文件路径所对应的文件存在，并且是一个文件，则直接删除
        if (file.exists() && file.isFile()) {
            if (file.delete()) {
                return true;
            } else {
                System.out.println("删除单个文件" + fileName + "失败！");
                return false;
            }
        } else {
            System.out.println("删除单个文件失败：" + fileName + "不存在！");
            return false;
        }
    }

    public static void renameFile(String path, String oldname, String newname) {
        //新的文件名和以前文件名不同时,才有必要进行重命名
        if (!oldname.equals(newname)) {
            File oldfile = new File(path + "\\" + oldname);
            File newfile = new File(path + "\\" + newname);
            if (!oldfile.exists()) {
                System.out.println("需重命名的文件不存在");
                //重命名文件不存在
                return;
            }
            //若在该目录下已经有一个文件和新文件名相同，则不允许重命名
            if (newfile.exists()) {
                System.out.println(newname + "已经存在！");
            } else {
                oldfile.renameTo(newfile);
            }
        } else {
            System.out.println("新文件名和旧文件名相同...");
        }
    }

    public static boolean createFile(String directoryName, String fileName) {
        File file1 = new File(directoryName);
        File dir = new File(file1, fileName);
        try {
            if (!dir.exists()) {
                if (dir.createNewFile())
                    return true;
                else {
                    return false;
                }
            } else {
                return false;
            }
        } catch (IOException e) {
            System.out.println("createFile:" + e.getMessage());
            return false;
        }
    }

    public ArrayList<HashMap<String, Object>> toMapList(List<Topic> topicList,String textPath) {
        ArrayList<HashMap<String, Object>> topicInfo = new ArrayList<>(15);
        //返回浏览信息集合，包括topic和对应的浏览次数
        ArrayList<HashMap<String, Object>> readCountList = this.showReadCount(textPath);
        for (int i = 0; i < topicList.size(); i++) {
            HashMap<String, Object> map1 = new HashMap<String, Object>();
            map1.put("readCount", getUniqueReadCount(readCountList, topicList.get(i).getTopicId()));
            map1.put("topicDetail", topicList.get(i).getTopicDetail());
            map1.put("topicId", topicList.get(i).getTopicId());
            map1.put("topicBelongCourseId", topicList.get(i).getBelongCourseId());
            map1.put("topicTime", topicList.get(i).getTopicTime());
            map1.put("topicWriterId", topicList.get(i).getTopicWritterId());
            map1.put("topicWriterType", topicList.get(i).getTopicWritterType());
            map1.put("reportNum", topicList.get(i).getReportNum());
            if (topicList.get(i).getTopicWritterType() == 1) {
                map1.put("writterType", "老师");
            } else {
                map1.put("writterType", "学生");
            }
            //如果是学生,查询学生表
            if (topicList.get(i).getTopicWritterType() == 0)
            {
                Student student = userDao.selectStudentById(topicList.get(i).getTopicWritterId());
                map1.put("topicWriterName", student.getStudentName());
                map1.put("topicWriterImg", student.getStudentPic());
            } else {
                Teacher teacher = teacherDao.selectByPrimaryKey(topicList.get(i).getTopicWritterId());
                map1.put("topicWriterName", teacher.getTeacherName());
                map1.put("topicWriterImg", teacher.getTeacherPic());
            }
            List<Comment> comList = comM.selectByTopicId(topicList.get(i).getTopicId());
            map1.put("comNum", comList.size());
            topicInfo.add(map1);
        }
        return topicInfo;
    }

    public Integer getUniqueReadCount(ArrayList<HashMap<String, Object>> readCountList, Integer topicId) {
        for (int i = 0; i < readCountList.size(); i++) {
            if (Integer.parseInt(readCountList.get(i).get("topicId").toString()) == topicId) {
                return Integer.parseInt(readCountList.get(i).get("readCount").toString());
            }
        }
        return -1;
    }
}
