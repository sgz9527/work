package com.xqq.service;

import com.xqq.pojo.Comment;
import com.xqq.pojo.Message;
import com.xqq.pojo.Praise_commentKey;
import com.xqq.pojo.Topic;

import javax.servlet.http.HttpSession;
import java.util.ArrayList;
import java.util.HashMap;

public interface CommentService {
    /**
     *显示评论
     * @param topicId
     * @param session
     * @return
     */
    public ArrayList<HashMap<String, Object>> showComment(Integer topicId,HttpSession session);

    /**
     *写评论
     * @param comm
     * @param session
     * @return
     */

    public HashMap<String, Object> writeComment(Comment comm,HttpSession session);

    /**
     *评论点赞
     * @param pra
     * @return
     */
    public int praise(Praise_commentKey pra);

    /**
     *显示帖子
     * @param courseId
     * @param dir
     * @return
     */

    public ArrayList<HashMap<String, Object>> showTopic(Integer courseId,String dir);

    /**
     *发表帖子
     * @param topic
     * @param dir
     * @return
     */

    public boolean writeTopic(Topic topic,String dir);

    /**
     *统计帖子浏览次数
     * @param topicId
     * @param dir
     * @return
     */

    public HashMap<String, Object> readCount(Integer topicId,String dir);

    /**
     *举报不良帖子
     * @param topicId
     * @return
     */

    public HashMap<String, Object> reportTopic(Integer topicId);

    /**
     *举报不良评论
     * @param commentId
     * @return
     */

    public HashMap<String, Object> reportComment(Integer commentId);

    /**
     *获取学生论坛活跃值
     * @param studentId
     * @return
     */

    public HashMap<String, Object> getActivity(Integer studentId);

    /**
     *显示举报评论列表，用于管理员审核
     * @return
     */

    ArrayList<HashMap<String,Object>> showCheckComment();

    /**
     * 禁止被举报评论的显示
     * @param msg
     * @return
     */

    HashMap<String,Object> forbidenComment(Message msg);

}
