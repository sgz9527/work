package com.xqq.dao;

import com.xqq.pojo.Comment;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository("CommentMapper")
public interface CommentMapper {
    int insert(Comment record);

    int insertSelective(Comment record);

    int deleteByPrimaryKey(Integer commentId);

    Comment selectByTimeAndWritter(@Param("commentWritterId") Integer commentWritterId, @Param("commentWritterType") short commentWritterType, @Param("commentTime") String commentTime);

    int updateByPrimaryKeySelective(Comment record);

    int updateByPrimaryKey(Comment record);

    int updateReportNum(@Param("commentId") Integer commentId);

    Comment selectByPrimaryKey(Integer commentId);
    Comment selectByPrimaryKey1(Integer commentId);

    ArrayList<Comment> selectByTopicId(@Param("topicId") Integer topicId);

    ArrayList<Comment> selectByStudentId(@Param("studentId") Integer studentId);

    ArrayList<Comment> selectReply(@Param("commentId") Integer commentId);

    ArrayList<Comment> selectByCourseId(Integer courseId);

    ArrayList<Comment> selectByReportNum(int maxReportNum);

    int updateByCommentId(Integer messageCommentId);

}