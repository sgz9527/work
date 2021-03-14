package com.xqq.controller;

import com.xqq.pojo.Comment;
import com.xqq.pojo.Message;
import com.xqq.pojo.Praise_commentKey;
import com.xqq.pojo.Topic;
import com.xqq.service.CommentService;
import com.xqq.staticmethod.getDate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

@Controller("commentAction")
public class commentAction {
    @Autowired
    CommentService commentServ;
    HashMap<String, Object> map = new HashMap<>();
    ArrayList<HashMap<String, Object>> listMap = new ArrayList<>();

    /**************评论区显示****************/
    @RequestMapping("showTopic")
    @ResponseBody
    public ArrayList<HashMap<String, Object>> showTopic(HttpServletRequest request,Integer courseId) {
        String textPath=request.getSession().getServletContext().getRealPath("/Texts");
        listMap = commentServ.showTopic(courseId,textPath);
        return listMap;
    }

    @RequestMapping("showComment")
   /**
    接口返回参数(list<map>)
            map.put("writterId",writterId);
            map.put("writterName",writterName);
            map.put("writterTime",writterTime);
            map.put("writterType",writterType);
            map.put("PraiseNum",PraiseNum);
            map.put("ReplyNum",ReplyNum);
            map.put("ReplyToId",replyToId);
            map.put("writterInfo",writterInfo);
    所需参数:topicId
    */
    @ResponseBody
    public ArrayList<HashMap<String, Object>> showComment(HttpServletRequest request,Integer topicId) {
        HttpSession mysession=request.getSession();
        listMap = commentServ.showComment(topicId,mysession);
        return listMap;
    }
    @RequestMapping("showCheckComment")
    /**
     接口返回参数(list<map>)
     map.put("commentWriterId",writterId);
     map.put("commentWriterName",writterName);
     map.put("commentTime",writterTime);
     map.put("commentInfo",writterInfo);
     commentId
     */
    @ResponseBody
    public ArrayList<HashMap<String, Object>> showCheckComment() {
        System.out.println("commentAction.showCheckComment");
        listMap = commentServ.showCheckComment();
        return listMap;
    }
    @RequestMapping("forbidenComment")

    @ResponseBody
    public ArrayList<HashMap<String, Object>> forbidenComment(Integer messageCommentId,Integer belongStudentId,Integer forbidenDay) {
        Date dateNow=new Date();
        Message msg=new Message();
        msg.setMessageTime(dateNow);
        msg.setMessageForbidenBeginTime(dateNow);
        msg.setMessageCommentId(messageCommentId);
        msg.setMessageStudentId(belongStudentId);
        Date dateEnd=new Date();
        dateEnd.setTime(((dateNow.getTime()/getDate.DAY)+forbidenDay)*getDate.DAY);
        msg.setMessageForbidenEndTime(dateEnd);
        map= commentServ.forbidenComment(msg);
        return listMap;
    }
    /**************评论显示******************/
    /***************写评论*******************/
    @RequestMapping("writeComment")
    /**
    返回参数：Map:isWrite
    所需参数：
     name="belongTopicId"
     name="commentInfo"
     name="writterId"
     name="commentTime"
     name="commentWritterType"
     */
    @ResponseBody
    public HashMap<String, Object> writeComment(HttpServletRequest request,Comment comm) {
        HttpSession mysession=request.getSession();
        map = commentServ.writeComment(comm,mysession);
        return map;
    }

    @RequestMapping("writeTopic")
    @ResponseBody
    public HashMap<String, Object> writeTopic(HttpServletRequest request,Topic topic) {
        String textPath=request.getSession().getServletContext().getRealPath("/Texts");
        if (commentServ.writeTopic(topic,textPath)) {
            map.put("isWrite", "ok");
        } else {
            map.put("isWrite", "no");
        }
        return map;
    }

    /*********************点赞功能******************/
    @RequestMapping("praise")
    @ResponseBody
    public HashMap<String, Object> praise(Praise_commentKey pra) {
        int result = commentServ.praise(pra);
        //点赞成功
        if (result == 0)
        {
            map.put("isPraise", "ok");
        } else if (result == 1) {
            map.put("isPraise", "alreadyPraise");
        } else {
            map.put("isPraise", "error");
        }
        return map;
    }

    @RequestMapping("readCount")
    @ResponseBody
    public HashMap<String, Object> ReadCount(HttpServletRequest request,Integer topicId) {//统计浏览次数
        String textPath=request.getSession().getServletContext().getRealPath("/Texts");
        map = commentServ.readCount(topicId,textPath);
        return map;
    }

    @RequestMapping("reportTopic")
    @ResponseBody
    public HashMap<String, Object> reportTopic(Integer topicId) {//统计浏览次数
        System.out.println("reportTopicId:" + topicId);
        map = commentServ.reportTopic(topicId);
        // System.out.println("readCountMapSize:"+map.size());
        return map;
    }

    @RequestMapping("reportComment")
    @ResponseBody
    public HashMap<String, Object> reportComment(Integer commentId) {//统计浏览次数
        System.out.println("reportCommentId:" + commentId);
        map = commentServ.reportComment(commentId);
        // System.out.println("readCountMapSize:"+map.size());
        return map;
    }

    @RequestMapping("activityValue")//计算活跃度
    @ResponseBody
    public HashMap<String, Object> activity(Integer studentId) {//统计浏览次数
        System.out.println("activity--studentId:" + studentId);
        map = commentServ.getActivity(studentId);//得到论坛活跃度
        // System.out.println("readCountMapSize:"+map.size());
        return map;
    }

}
