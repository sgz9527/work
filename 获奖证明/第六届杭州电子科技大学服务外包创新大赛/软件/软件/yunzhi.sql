/*
Navicat MySQL Data Transfer

Source Server         : wwwbccom
Source Server Version : 50521
Source Host           : localhost:3306
Source Database       : yunzhi

Target Server Type    : MYSQL
Target Server Version : 50521
File Encoding         : 65001

Date: 2019-06-10 18:29:58
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `chapter`
-- ----------------------------
DROP TABLE IF EXISTS `chapter`;
CREATE TABLE `chapter` (
  `chapterId` int(11) NOT NULL AUTO_INCREMENT,
  `chapterTitle` varchar(20) NOT NULL,
  `chapterName` varchar(50) NOT NULL,
  `belongCourseId` int(6) NOT NULL,
  `chapterOrder` int(3) NOT NULL,
  PRIMARY KEY (`chapterId`),
  KEY `FK_ID10` (`belongCourseId`),
  CONSTRAINT `FK_ID10` FOREIGN KEY (`belongCourseId`) REFERENCES `course` (`courseId`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=54 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of chapter
-- ----------------------------
INSERT INTO `chapter` VALUES ('1', '第四单元', '让我们认识这个世界', '5', '4');
INSERT INTO `chapter` VALUES ('2', '第三单元', '探索海洋生物', '5', '3');
INSERT INTO `chapter` VALUES ('3', '第二单元', '走进非洲草原', '5', '2');
INSERT INTO `chapter` VALUES ('4', '第一单元', '发现世界奥秘', '5', '1');
INSERT INTO `chapter` VALUES ('5', '第四单元', '发现世界奥秘', '1', '4');
INSERT INTO `chapter` VALUES ('6', '第三单元', '走进非洲草原', '1', '3');
INSERT INTO `chapter` VALUES ('7', '第二单元', '探索海洋生物', '1', '2');
INSERT INTO `chapter` VALUES ('8', '第一单元', '让我们认识这个世界', '1', '1');
INSERT INTO `chapter` VALUES ('9', '第四单元', '发现世界奥秘', '2', '4');
INSERT INTO `chapter` VALUES ('10', '第三单元', '走进非洲草原', '2', '3');
INSERT INTO `chapter` VALUES ('11', '第二单元', '探索海洋生物', '2', '2');
INSERT INTO `chapter` VALUES ('12', '第一单元', '让我们认识这个世界', '2', '1');
INSERT INTO `chapter` VALUES ('13', '第四单元', '发现世界奥秘', '3', '4');
INSERT INTO `chapter` VALUES ('14', '第三单元', '走进非洲草原', '3', '3');
INSERT INTO `chapter` VALUES ('15', '第二单元', '探索海洋生物', '3', '2');
INSERT INTO `chapter` VALUES ('16', '第一单元', '让我们认识这个世界', '3', '1');
INSERT INTO `chapter` VALUES ('17', '第四单元', '发现世界奥秘', '4', '4');
INSERT INTO `chapter` VALUES ('18', '第三单元', '走进非洲草原', '4', '3');
INSERT INTO `chapter` VALUES ('19', '第二单元', '探索海洋生物', '4', '2');
INSERT INTO `chapter` VALUES ('20', '第一单元', '让我们认识这个世界', '4', '1');
INSERT INTO `chapter` VALUES ('25', '第四单元', '发现世界奥秘', '7', '4');
INSERT INTO `chapter` VALUES ('26', '第三单元', '走进非洲草原', '7', '3');
INSERT INTO `chapter` VALUES ('27', '第二单元', '探索海洋生物', '7', '2');
INSERT INTO `chapter` VALUES ('28', '第一单元', '让我们认识这个世界', '7', '1');
INSERT INTO `chapter` VALUES ('29', '第四单元', '发现世界奥秘', '8', '4');
INSERT INTO `chapter` VALUES ('30', '第三单元', '走进非洲草原', '8', '3');
INSERT INTO `chapter` VALUES ('31', '第二单元', '探索海洋生物', '8', '2');
INSERT INTO `chapter` VALUES ('32', '第一单元', '让我们认识这个世界', '8', '1');
INSERT INTO `chapter` VALUES ('33', '第四单元', '发现世界奥秘', '9', '4');
INSERT INTO `chapter` VALUES ('34', '第三单元', '走进非洲草原', '9', '3');
INSERT INTO `chapter` VALUES ('35', '第二单元', '探索海洋生物', '9', '2');
INSERT INTO `chapter` VALUES ('36', '第一单元', '让我们认识这个世界', '9', '1');
INSERT INTO `chapter` VALUES ('37', '第四单元', '发现世界奥秘', '10', '4');
INSERT INTO `chapter` VALUES ('38', '第三单元', '走进非洲草原', '10', '3');
INSERT INTO `chapter` VALUES ('39', '第二单元', '探索海洋生物', '10', '2');
INSERT INTO `chapter` VALUES ('40', '第一单元', '让我们认识这个世界', '10', '1');
INSERT INTO `chapter` VALUES ('41', '第四单元', '发现世界奥秘', '11', '4');
INSERT INTO `chapter` VALUES ('42', '第三单元', '走进非洲草原', '11', '3');
INSERT INTO `chapter` VALUES ('43', '第二单元', '探索海洋生物', '11', '2');
INSERT INTO `chapter` VALUES ('44', '第一单元', '让我们认识这个世界', '11', '1');
INSERT INTO `chapter` VALUES ('45', '第四单元', '发现世界奥秘', '12', '4');
INSERT INTO `chapter` VALUES ('46', '第三单元', '走进非洲草原', '12', '3');
INSERT INTO `chapter` VALUES ('47', '第二单元', '探索海洋生物', '12', '2');
INSERT INTO `chapter` VALUES ('48', '第一单元', '让我们认识这个世界', '12', '1');
INSERT INTO `chapter` VALUES ('49', '第三单元', '发现世界奥秘', '13', '3');
INSERT INTO `chapter` VALUES ('50', '第二单元', '探索海洋生物', '13', '2');
INSERT INTO `chapter` VALUES ('51', '第一单元', '让我们认识这个世界', '13', '1');
INSERT INTO `chapter` VALUES ('52', '第一单元', '我的电竞生涯', '150', '1');
INSERT INTO `chapter` VALUES ('53', '第二单元', '主播生涯', '150', '2');

-- ----------------------------
-- Table structure for `checkcourse`
-- ----------------------------
DROP TABLE IF EXISTS `checkcourse`;
CREATE TABLE `checkcourse` (
  `checkId` int(8) NOT NULL AUTO_INCREMENT,
  `courseName` varchar(30) NOT NULL,
  `courseIntroduce` varchar(200) NOT NULL,
  `introduceSrc` varchar(60) NOT NULL,
  `courseType` varchar(20) NOT NULL,
  `teacherId` int(8) NOT NULL,
  `belongSchId` int(8) NOT NULL,
  `checkState` varchar(15) NOT NULL DEFAULT 'waitCheck' COMMENT '总共有 待审核，已通过，失败三种状态',
  `teacherPhone` varchar(20) NOT NULL,
  `posterSrc` varchar(60) NOT NULL,
  `suplement` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`checkId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of checkcourse
-- ----------------------------

-- ----------------------------
-- Table structure for `comment`
-- ----------------------------
DROP TABLE IF EXISTS `comment`;
CREATE TABLE `comment` (
  `commentId` int(6) NOT NULL AUTO_INCREMENT,
  `commentTime` varchar(20) NOT NULL,
  `commentInfo` varchar(200) NOT NULL,
  `commentWritterId` int(6) NOT NULL,
  `commentWritterType` smallint(1) NOT NULL DEFAULT '0',
  `replyToId` int(6) NOT NULL DEFAULT '-1' COMMENT '默认非回复类型',
  `belongTopicId` int(6) NOT NULL,
  `reportNum` int(3) NOT NULL DEFAULT '0',
  `isForbiden` smallint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`commentId`),
  KEY `FK_ID` (`belongTopicId`),
  CONSTRAINT `FK_ID` FOREIGN KEY (`belongTopicId`) REFERENCES `topic` (`topicId`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=105 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of comment
-- ----------------------------
INSERT INTO `comment` VALUES ('1', '2019-01-02 13:56:23', '这门课程真的是太好了，这门课你值得一看', '1', '0', '-1', '1', '2', '0');
INSERT INTO `comment` VALUES ('4', '2019-01-05 22:57:46', '呀呀呀呀呀呀晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕晕', '3', '1', '-1', '2', '0', '0');
INSERT INTO `comment` VALUES ('5', '2019-01-05 22:58:40', '23333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333', '3', '1', '-1', '1', '1', '0');
INSERT INTO `comment` VALUES ('15', '2019-02-03 13:25:12', '竟然没有人来评论我的话题？太失败了。', '2', '0', '-1', '3', '0', '0');
INSERT INTO `comment` VALUES ('17', '2019-02-03 14:20:29', '这个问题很难回答', '4', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('20', '2019-02-04 08:34:14', '？？？', '2', '0', '-1', '4', '0', '0');
INSERT INTO `comment` VALUES ('22', '2019-02-05 18:10:48', 'this course is so important that you can not ignore this course.', '2', '0', '-1', '8', '4', '1');
INSERT INTO `comment` VALUES ('24', '2019-02-05 19:33:16', '这门课到底多少平时分？没一个人来回复我？？？', '2', '0', '-1', '9', '1', '0');
INSERT INTO `comment` VALUES ('28', '2019-02-07 12:32:25', '少年强则国强，人民有信仰，民族有力量，中国有希望', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('30', '2019-02-08 09:01:34', '你可真是个机灵鬼', '3', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('38', '2019-02-12 10:40:34', '[鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('39', '2019-02-12 10:58:41', '[鄙视]', '2', '0', '-1', '2', '0', '0');
INSERT INTO `comment` VALUES ('42', '2019-02-12 11:27:13', '[awuehi', '2', '0', '-1', '2', '5', '1');
INSERT INTO `comment` VALUES ('47', '2019-02-12 12:33:16', '[[[鄙视]]]]', '2', '0', '-1', '8', '0', '0');
INSERT INTO `comment` VALUES ('50', '2019-02-12 16:31:35', '[鄙视]', '2', '0', '-1', '1', '2', '0');
INSERT INTO `comment` VALUES ('51', '2019-02-12 16:33:52', 'rtuyrti7ui67[鄙视]', '2', '0', '-1', '1', '7', '1');
INSERT INTO `comment` VALUES ('52', '2019-02-12 16:38:55', '[鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('53', '2019-02-12 16:45:35', '[鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视]', '2', '0', '-1', '1', '2', '0');
INSERT INTO `comment` VALUES ('64', '2019-02-12 19:00:47', '[鄙视][鄙视]噗噗噗噗噗', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('68', '2019-02-12 19:10:01', '一月又一月[鄙视][鄙视][鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('69', '2019-02-12 19:13:51', '呵呵 哒哒哒[鄙视]安全问题外而过', '2', '0', '-1', '3', '0', '0');
INSERT INTO `comment` VALUES ('70', '2019-02-12 19:14:06', '呵呵 哒哒哒[鄙视]', '2', '0', '-1', '3', '0', '0');
INSERT INTO `comment` VALUES ('71', '2019-02-12 19:15:51', '[鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('72', '2019-02-12 19:16:39', '[鄙视]', '2', '0', '-1', '4', '0', '0');
INSERT INTO `comment` VALUES ('73', '2019-02-12 19:17:15', '[鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视][鄙视]', '2', '0', '-1', '4', '0', '0');
INSERT INTO `comment` VALUES ('82', '2019-02-14 09:28:28', '[鄙视]', '2', '0', '-1', '1', '4', '1');
INSERT INTO `comment` VALUES ('83', '2019-04-04 12:46:14', '[鄙视]', '2', '0', '-1', '1', '4', '1');
INSERT INTO `comment` VALUES ('85', '2019-04-06 19:07:52', 'i want to say something', '1', '0', '-1', '12', '1', '0');
INSERT INTO `comment` VALUES ('86', '2019-04-16 13:24:43', '[鄙视]', '1', '1', '-1', '9', '3', '0');
INSERT INTO `comment` VALUES ('87', '2019-04-16 13:25:31', '？？？', '1', '1', '-1', '9', '3', '0');
INSERT INTO `comment` VALUES ('88', '2019-04-16 13:57:44', '222', '1', '1', '-1', '9', '5', '0');
INSERT INTO `comment` VALUES ('89', '2019-05-09 12:47:32', '已举报，坚决维护讨论区和谐发言！', '1', '0', '51', '1', '0', '0');
INSERT INTO `comment` VALUES ('90', '2019-05-09 15:12:11', '5849851654', '1', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('91', '2019-05-12 15:01:42', '1234', '1', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('92', '2019-06-04 20:08:31', 'awoijhroi[鄙视]awhtegfbsdwr[鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('93', '2019-06-04 20:11:22', 'iaehrihug', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('94', '2019-06-04 20:11:29', '[鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('95', '2019-06-04 20:11:37', '[鄙视][鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('96', '2019-06-04 20:11:53', '[鄙视]etjgdjmryj[鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('97', '2019-06-04 20:12:03', 'ahtehrfh[鄙视][鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('98', '2019-06-04 20:12:17', '[鄙视][鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('99', '2019-06-04 20:12:31', 'aerthgedfbgsdvr[鄙视][鄙视][鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('100', '2019-06-04 20:41:31', '[鄙视]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('101', '2019-06-04 21:48:33', '[滑稽][发怒]<img style=\"width:25px;height:25px;\" src=\"img/konghuang.png\">[开心][微笑][爱心][调皮][苦笑][爱心][调皮][微笑][微笑][微笑]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('102', '2019-06-04 21:49:41', '<img style=\"width:25px;height:25px;\" src=\"img/konghuang.png\">', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('103', '2019-06-04 21:49:58', '<img style=\"width:25px;height:25px;\" src=\"img/konghuang.png\">[笑哭][笑哭]', '2', '0', '-1', '1', '0', '0');
INSERT INTO `comment` VALUES ('104', '2019-06-04 22:09:32', '<img style=\"width:25px;height:25px;\" src=\"img/konghuang.png\">[笑哭][笑哭]', '2', '1', '-1', '1', '0', '0');

-- ----------------------------
-- Table structure for `course`
-- ----------------------------
DROP TABLE IF EXISTS `course`;
CREATE TABLE `course` (
  `courseId` int(11) NOT NULL AUTO_INCREMENT,
  `courseName` varchar(30) NOT NULL,
  `courseIntroduce` varchar(300) NOT NULL,
  `coursePostSrc` varchar(50) NOT NULL,
  `courseProgress` int(3) NOT NULL DEFAULT '0',
  `belongSchId` int(3) NOT NULL DEFAULT '1',
  `courseType` varchar(30) NOT NULL DEFAULT '',
  `introduceVideoSrc` varchar(50) NOT NULL DEFAULT '',
  `evaluationLevel` varchar(300) NOT NULL DEFAULT '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。',
  `evaluationExamWeight` double(3,2) NOT NULL DEFAULT '0.40',
  `evaluationChatWeight` double(3,2) NOT NULL DEFAULT '0.20',
  `evaluationTestWeight` double(3,2) NOT NULL DEFAULT '0.40',
  `examSuplement` varchar(155) DEFAULT NULL,
  `examLimitTime` double(3,1) DEFAULT '60.0',
  `examOneTitle` varchar(255) DEFAULT NULL,
  `examOneAnswer` varchar(2000) DEFAULT NULL,
  `examTwoTitle` varchar(255) DEFAULT NULL,
  `examTwoAnswer` varchar(2000) DEFAULT NULL,
  `examThreeTitle` varchar(255) DEFAULT NULL,
  `examThreeAnswer` varchar(2000) DEFAULT NULL,
  `examFourTitle` varchar(255) DEFAULT NULL,
  `examFourAnswer` varchar(2000) DEFAULT NULL,
  `examFiveTitle` varchar(255) DEFAULT NULL,
  `examFiveAnswer` varchar(2000) DEFAULT NULL,
  `examPublishTeacherId` int(8) NOT NULL DEFAULT '-1' COMMENT '-1表示暂无exam信息发布',
  `examPass` tinyint(1) NOT NULL DEFAULT '-1' COMMENT '-1为待审核，0为不通过，1为通过',
  `examStartTime` date DEFAULT '2019-04-01',
  `examEndTime` date DEFAULT '2019-06-30',
  PRIMARY KEY (`courseId`),
  UNIQUE KEY `courseName` (`courseName`)
) ENGINE=InnoDB AUTO_INCREMENT=151 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of course
-- ----------------------------
INSERT INTO `course` VALUES ('1', '计算机组成原理', '计算机组成原理是本科计算机科学与技术专业的学科基础课，重点讲授计算机系统的硬件组成，及其主要功能子系统的基本原理和逻辑设计，内容安排遵照本科教学大纲，兼顾硕士生入学专业课考试，有助于培养学习者计算机硬件系统的分析和设计能力，给参加与本课程相关考试的学习者插上翅膀，如虎添翼。', '1.jpg', '3', '4', '工科|计算机|硬件理论', 'video1.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', '1111111111111111111111', '60.0', '1111', '1111111111111111', '22222222222222', '222222222222222', '3333333', '333333333333333333333333333333333333', '4444444444444444444444', '4444444444444444444444444444444444', '555555555', '5555555555555555555555555555555', '2', '1', '2019-05-29', '2019-06-07');
INSERT INTO `course` VALUES ('2', '数据结构', '程序=数据结构+算法，利用数据结构的知识设计“好”的算法是程序猿的基本要求，许多IT企业的面试和笔试都广泛地涉及各种各样的数据结构。通过数据结构课程的学习，充分理解这些数据结构，并掌握数据结构的数据组织和数据处理方法，你也会成为编程“牛人”。', '2.jpg', '3', '2', '工科|计算机|算法', 'mov_bbb.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', '本次测试是本门课程期末测试，请同学们认真按规定时间内完成，在时间截止前点提交测试，仅有一次提交机会，诚信考试，杜绝舞弊', '60.0', '二叉树后序遍历非递归算法C/C++实现，关键代码不得使用伪代码代替。', '/**\r\n*returnSize:用于返回结果数组的大小\r\n        ---结果用自定义的*result数组存储，它的长度传给*returnSize\r\n**/\r\nint* postorderTraversal(struct TreeNode* root, int* returnSize)\r\n{\r\n    if( root == NULL ) return NULL;\r\n \r\n    /**自定义结果数组并初始化**/\r\n    int* result = (int*)malloc(1000*sizeof(int));\r\n    int len = 0;\r\n \r\n    /**定义栈并初始化**/\r\n    Sq_stack* stk_result = (Sq_stack*)malloc(sizeof(Sq_stack));\r\n    initStack(stk_result);\r\n    Sq_stack* temp_stk = (Sq_stack*)malloc(sizeof(Sq_stack));\r\n    initStack(temp_stk);\r\n \r\n    while( root || !isEmptyStack(temp_stk) )\r\n    {\r\n        /**将当前结点同时入临时栈和结果栈【根】，并遍历右子树----【右】**/\r\n        while( root )\r\n        {\r\n            Push(temp_stk,root);\r\n            Push(stk_result,root);\r\n            root = root->right;\r\n        }\r\n        /**当右子树遍历结束后，弹出临时栈栈顶结点，并遍历其左子树----【左】，继续while**/\r\n        if( !isEmptyStack(temp_stk) )\r\n        {\r\n            root = Pop(temp_stk);\r\n            root = root->left;\r\n        }\r\n    }\r\n    /**\r\n        当所有结点都访问完，即最后访问的树节点为空且临时栈也为空时，\r\n        主算法结束，依次pop出结果栈结点\r\n    **/\r\n    struct TreeNode* e = (struct TreeNode*)malloc(sizeof(struct TreeNode));\r\n    while( !isEmptyStack(stk_result) )\r\n    {\r\n        e = Pop(stk_result);\r\n        result[len++] = e->val;\r\n    }\r\n    free(e);\r\n    *returnSize = len;\r\n    return result;', '简述你本学期在数据结构课程上的收获。', '略', '对（23,24,67,85,69,24,39,45）进行堆排序。', '具体参考数据结构书，严蔚敏版堆排序过程', '描述最短路径的迪杰斯特拉算法', '具体算法见数据结构书严蔚敏版最短路径算法实现。', '对链表逆序算法的C/C++代码实现。', 'void destoryList(PNode* list)\r\n{\r\n	if(NULL == *list){\r\n		return;\r\n	}\r\n	PNode node = *list;\r\n	PNode tmp = NULL;\r\n	while(NULL != node)\r\n	{\r\n		tmp = node;\r\n		node->next = NULL;\r\n		free(node);\r\n		node = tmp->next;\r\n	}\r\n	*list = NULL;\r\n}\r\n \r\nbool isEmpty(PNode list)\r\n{\r\n	if(NULL == list){\r\n		return true;\r\n	}\r\n	else {\r\n		return false;\r\n	}\r\n}\r\n \r\nvoid insert(PNode head,PNode p,PNode s)\r\n{\r\n	if(NULL == head){\r\n		return;\r\n	}\r\n	if(p == head){\r\n		s->next = head->next;\r\n		head->next = s;\r\n	//	head->data = s->data;\r\n	}\r\n	else{\r\n		s->next = p->next;\r\n		p->next = s;\r\n	}\r\n}\r\n \r\nPNode makenode(T i)\r\n{\r\n	PNode p = (PNode)malloc(sizeof(List));\r\n	p->data = i;\r\n	p->next = NULL;\r\n	return p;\r\n}', '2', '1', '2019-05-29', '2019-06-10');
INSERT INTO `course` VALUES ('3', '探索土星上的生命', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '3.jpg', '3', '2', '理科|地理|天文', 'mov_bbb.mp4', '完成所有课程内容的学习。\r\n在此基础上，最后成绩的构成方式：\r\n单元测验占30%，期末考试占总成绩的50%，论坛表现占20%。分数按百分制计，60分以上为及格分，80分以上为优秀分。\r\n论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！\r\n期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。\r\n课程结束后得到本门课考试分数，并颁发“通过”和“优秀”两种证书。', '0.50', '0.20', '0.30', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('4', 'M78星云探索发现', '这是一门很有意思的课程，亲，你知道保卫地球的奥特曼是从哪里来吗？没错，就是从伟大的M78星云而来，而本门课程将带领小伙伴们共同探究M78星云的奥密，让凹凸曼们不再神秘，欢迎小伙伴们加入我们的探索旅程，一起欢乐一起分享！', '4.jpg', '4', '4', '理科|地理|天文', 'video1.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('5', '探索发现', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '5.jpg', '3', '2', '理科|地理|水土', 'mov_bbb.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('7', '日语探讨', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '7.jpg', '3', '2', '文科|语言|日语', 'mov_bbb.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('8', '汇编语言', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '8.jpg', '4', '2', '工科|计算机|编程语言', 'video1.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('9', '编译原理', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '9.jpg', '3', '2', '工科|计算机|硬件理论', 'video1.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('10', '操作系统', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '10.jpg', '4', '2', '工科|计算机|硬件理论', 'mov_bbb.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('11', '数字电路', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '11.jpg', '3', '2', '工科|计算机|硬件理论', 'video1.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('12', '模拟电子技术基础', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '12.jpg', '4', '2', '工科|电子信息|电路基础', 'mov_bbb.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('13', '电子发展史', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '13.jpg', '3', '1', '工科|电子信息|电子概论', 'mov_bbb.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('14', '台海局势', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '13.jpg', '4', '6', '文科|政治|国际形势', 'mov_bbb.mp4 ', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('15', '中美对弈', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '13.jpg', '3', '6', '文科|政治|国际形势', 'video1.mp4 ', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('16', '中日外交', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '13.jpg', '4', '6', '文科|政治|国际形势', 'mov_bbb.mp4 ', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('17', '南海争端', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '13.jpg', '3', '1', '文科|政治|国际形势', 'mov_bbb.mp4 ', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('18', '叙利亚危机', '这是一门全新的课程,只需一分钟你介会像我一样爱像介门课程，介们课程让我们了解到什么是土星，什么是生命，什么是土星上的生命，土星上的生命从哪里来，又到哪里去，让我们期待奇迹的到来。', '13.jpg', '4', '1', '文科|政治|国际形势', 'video1.mp4 ', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('21', '高等数学', '这是一门很有意思的理科数学课程，你肯定会喜欢这门课程', '599112DA2A5A0D74CCDA98F75F341005.jpg', '3', '1', '理科|数学|高等数学', 'video1.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('22', '马克思主义原理', 'aiuuhwjuighuiwajefhoiu', '13.jpg', '4', '1', '文科|政治|思想理论', 'video1.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('23', '中国礼仪文化', 'oaijwiojgoijosidjaoiwjergoijeroigjeoir', '13.jpg', '3', '1', '文科|其他', 'mov_bbb.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('100', '课程0', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|语言|日语', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('101', '课程1', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('102', '课程2', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('103', '课程3', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('104', '课程4', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('105', '课程5', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('106', '课程6', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('107', '课程7', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('108', '课程8', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|电子信息|电路基础', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('109', '课程9', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('110', '课程10', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('111', '课程11', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|语言|日语', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('112', '课程12', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|电子信息|电路基础', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('113', '课程13', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('114', '课程14', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|数学|大学数学', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('115', '课程15', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('116', '课程16', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|数学|大学数学', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('117', '课程17', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('118', '课程18', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('119', '课程19', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('120', '课程20', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('121', '课程21', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|语言|日语', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('122', '课程22', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|数学|大学数学', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('123', '课程23', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('124', '课程24', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('125', '课程25', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('126', '课程26', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('127', '课程27', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|电子信息|电路基础', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('128', '课程28', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('129', '课程29', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|数学|大学数学', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('130', '课程30', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('131', '课程31', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|数学|大学数学', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('132', '课程32', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|语言|日语', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('133', '课程33', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('134', '课程34', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('135', '课程35', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|电子信息|电路基础', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('136', '课程36', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('137', '课程37', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|电子信息|电路基础', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('138', '课程38', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('139', '课程39', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('140', '课程40', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('141', '课程41', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|电子信息|电路基础', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('142', '课程42', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|电子信息|电路基础', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('143', '课程43', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('144', '课程44', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|语言|日语', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('145', '课程45', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('146', '课程46', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('147', '课程47', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '理科|地理|水土', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('148', '课程48', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '工科|计算机|编程语言', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('149', '课程49', 'aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf', '13.jpg', '0', '2', '文科|政治|思想理论', 'video1.mp4', 'ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');
INSERT INTO `course` VALUES ('150', '我的电竞人生', '本课程主要讲述卢本伟老师的传奇人生以及电竞理论常识', 'Uploadhuangzu.jpeg', '1', '1', '文科|其他', 'Uploadvideo11.mp4', '完成所有课程内容的学习。<br/>在此基础上，最后成绩的构成方式：<br/>单元测验占20%，期末考试占总成绩的40%，论坛表现占40%。分数按百分制计，60分以上为及格分，85分以上为优秀分。<br/>论坛表现是指学习者和其它学习者的交流情况，按论坛活跃度算，要求发帖或回帖的数量超过10个，多发高质量帖子有加分哦！<br/>期末考试为主观题，需要学习者完成书面写作作业，采取同伴互评的方式。<br/>课程结束后颁发“通过”和“优秀”两种证书。', '0.40', '0.20', '0.40', null, '60.0', null, null, null, null, null, null, null, null, null, null, '-1', '-1', '2019-04-01', '2019-06-30');

-- ----------------------------
-- Table structure for `exam_score`
-- ----------------------------
DROP TABLE IF EXISTS `exam_score`;
CREATE TABLE `exam_score` (
  `examId` int(11) NOT NULL AUTO_INCREMENT,
  `studentId` int(8) NOT NULL,
  `courseId` int(8) NOT NULL,
  `score` float(3,0) NOT NULL,
  PRIMARY KEY (`examId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of exam_score
-- ----------------------------

-- ----------------------------
-- Table structure for `manager`
-- ----------------------------
DROP TABLE IF EXISTS `manager`;
CREATE TABLE `manager` (
  `ManagerId` int(3) NOT NULL,
  `account` varchar(30) NOT NULL,
  `password` varchar(30) NOT NULL,
  `ManagerName` varchar(30) NOT NULL,
  PRIMARY KEY (`ManagerId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of manager
-- ----------------------------
INSERT INTO `manager` VALUES ('1', '123456xqq1', '666666', '这货不是管理员');

-- ----------------------------
-- Table structure for `message`
-- ----------------------------
DROP TABLE IF EXISTS `message`;
CREATE TABLE `message` (
  `messageId` int(8) NOT NULL AUTO_INCREMENT,
  `messageType` smallint(1) NOT NULL,
  `messageCourseId` int(8) DEFAULT NULL,
  `messageTopicId` int(8) DEFAULT NULL,
  `messageCommentId` int(8) DEFAULT NULL,
  `messageTestId` int(8) DEFAULT NULL,
  `messageExamId` int(8) DEFAULT NULL,
  `messageStudentId` int(8) DEFAULT NULL,
  `messageForbidenBeginTime` datetime DEFAULT NULL,
  `messageForbidenEndTime` datetime DEFAULT NULL,
  `messageExcerciseId` int(8) DEFAULT NULL,
  `messageTime` datetime NOT NULL,
  PRIMARY KEY (`messageId`)
) ENGINE=InnoDB AUTO_INCREMENT=31 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of message
-- ----------------------------
INSERT INTO `message` VALUES ('1', '1', '150', null, null, null, null, null, null, null, null, '2019-05-07 20:44:14');
INSERT INTO `message` VALUES ('9', '9', '1', null, '22', null, null, '2', '2019-05-08 21:41:41', '2019-05-10 08:00:00', null, '2019-05-08 21:41:41');
INSERT INTO `message` VALUES ('10', '9', '1', null, '42', null, null, '2', '2019-05-08 21:41:47', '2019-05-10 08:00:00', null, '2019-05-08 21:41:47');
INSERT INTO `message` VALUES ('11', '9', '1', null, '82', null, null, '2', '2019-05-08 21:41:49', '2019-05-10 08:00:00', null, '2019-05-08 21:41:49');
INSERT INTO `message` VALUES ('12', '9', '1', null, '83', null, null, '2', '2019-05-08 21:41:51', '2019-05-10 08:00:00', null, '2019-05-08 21:41:51');
INSERT INTO `message` VALUES ('13', '4', '1', '1', '89', null, null, null, null, null, null, '2019-05-09 12:47:56');
INSERT INTO `message` VALUES ('14', '9', '1', null, '51', null, null, '2', '2019-05-09 12:48:48', '2019-05-12 08:00:00', null, '2019-05-09 12:48:48');
INSERT INTO `message` VALUES ('15', '5', '1', '1', '90', null, null, '-1', null, null, null, '2019-05-09 15:12:11');
INSERT INTO `message` VALUES ('16', '5', '1', '1', '91', null, null, '-1', null, null, null, '2019-05-12 15:01:43');
INSERT INTO `message` VALUES ('17', '5', '1', '1', '92', null, null, '-1', null, null, null, '2019-06-04 20:08:31');
INSERT INTO `message` VALUES ('18', '5', '1', '1', '93', null, null, '-1', null, null, null, '2019-06-04 20:11:22');
INSERT INTO `message` VALUES ('19', '5', '1', '1', '94', null, null, '-1', null, null, null, '2019-06-04 20:11:29');
INSERT INTO `message` VALUES ('20', '5', '1', '1', '95', null, null, '-1', null, null, null, '2019-06-04 20:11:37');
INSERT INTO `message` VALUES ('21', '5', '1', '1', '96', null, null, '-1', null, null, null, '2019-06-04 20:11:53');
INSERT INTO `message` VALUES ('22', '5', '1', '1', '97', null, null, '-1', null, null, null, '2019-06-04 20:12:03');
INSERT INTO `message` VALUES ('23', '5', '1', '1', '98', null, null, '-1', null, null, null, '2019-06-04 20:12:17');
INSERT INTO `message` VALUES ('24', '5', '1', '1', '99', null, null, '-1', null, null, null, '2019-06-04 20:12:31');
INSERT INTO `message` VALUES ('25', '5', '1', '1', '100', null, null, '-1', null, null, null, '2019-06-04 20:41:31');
INSERT INTO `message` VALUES ('26', '5', '1', '1', '101', null, null, '-1', null, null, null, '2019-06-04 21:48:33');
INSERT INTO `message` VALUES ('27', '5', '1', '1', '102', null, null, '-1', null, null, null, '2019-06-04 21:49:41');
INSERT INTO `message` VALUES ('28', '5', '1', '1', '103', null, null, '-1', null, null, null, '2019-06-04 21:49:58');
INSERT INTO `message` VALUES ('29', '5', '1', '1', '104', null, null, '-1', null, null, null, '2019-06-04 22:09:32');
INSERT INTO `message` VALUES ('30', '7', '1', null, null, null, '1', '-1', null, null, null, '2019-06-05 12:02:33');

-- ----------------------------
-- Table structure for `message_stu`
-- ----------------------------
DROP TABLE IF EXISTS `message_stu`;
CREATE TABLE `message_stu` (
  `messageId` int(9) NOT NULL,
  `studentId` int(8) NOT NULL,
  PRIMARY KEY (`messageId`,`studentId`),
  KEY `FK_ID4` (`studentId`),
  CONSTRAINT `FK_ID3` FOREIGN KEY (`messageId`) REFERENCES `message` (`messageId`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK_ID4` FOREIGN KEY (`studentId`) REFERENCES `student` (`studentId`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of message_stu
-- ----------------------------
INSERT INTO `message_stu` VALUES ('1', '1');
INSERT INTO `message_stu` VALUES ('13', '1');
INSERT INTO `message_stu` VALUES ('30', '1');
INSERT INTO `message_stu` VALUES ('9', '2');
INSERT INTO `message_stu` VALUES ('10', '2');
INSERT INTO `message_stu` VALUES ('11', '2');
INSERT INTO `message_stu` VALUES ('12', '2');
INSERT INTO `message_stu` VALUES ('13', '2');
INSERT INTO `message_stu` VALUES ('14', '2');

-- ----------------------------
-- Table structure for `notice`
-- ----------------------------
DROP TABLE IF EXISTS `notice`;
CREATE TABLE `notice` (
  `noticeId` int(11) NOT NULL AUTO_INCREMENT,
  `noticeTitle` varchar(30) NOT NULL,
  `noticeDetail` varchar(300) NOT NULL,
  `belongCourseId` int(5) NOT NULL,
  `writer` varchar(30) NOT NULL,
  `time` varchar(30) NOT NULL,
  `systemTime` time NOT NULL,
  `noticeState` varchar(15) NOT NULL DEFAULT 'wait',
  PRIMARY KEY (`noticeId`)
) ENGINE=InnoDB AUTO_INCREMENT=22 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of notice
-- ----------------------------
INSERT INTO `notice` VALUES ('1', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '5', '二哈', '2019-4-13', '05:20:12', 'wait');
INSERT INTO `notice` VALUES ('2', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '5', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('3', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '1', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('4', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '1', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('5', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '2', '二哈', '2019-4-13', '05:20:12', 'wait');
INSERT INTO `notice` VALUES ('6', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '2', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('7', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '3', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('9', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '4', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('10', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '6', '二哈', '2019-4-13', '05:20:12', 'wait');
INSERT INTO `notice` VALUES ('11', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '6', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('12', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '7', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('13', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '8', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('14', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '9', '二哈', '2019-4-13', '05:20:12', 'wait');
INSERT INTO `notice` VALUES ('15', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '10', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('16', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '11', '二哈', '2019-4-13', '05:20:12', 'wait');
INSERT INTO `notice` VALUES ('17', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '11', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('18', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '12', '二哈', '2019-4-13', '05:20:12', 'wait');
INSERT INTO `notice` VALUES ('19', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '12', '二哈', '2019-4-13', '05:20:12', 'finish');
INSERT INTO `notice` VALUES ('20', '全国大学生创新创业大赛开启', 'awefgvrvdrearahwroghihfahvrguhguehuvahuerguivaur', '13', '二哈', '2019-4-13', '05:20:12', 'wait');
INSERT INTO `notice` VALUES ('21', '全国英雄联盟争霸赛开启', '杰拉网咖决定于2019年5月1日召开全国英雄联盟争霸大赛，欢迎各位召唤师们踊跃报名\n地址：杰拉网咖高沙文苑路店。要求段位在钻石及以上，又组建过战队经验者优先，\n奖金：50000RMB', '3', '卢本伟', '2019-4-13', '16:10:25', 'finish');

-- ----------------------------
-- Table structure for `praise_comment`
-- ----------------------------
DROP TABLE IF EXISTS `praise_comment`;
CREATE TABLE `praise_comment` (
  `commentId` int(6) NOT NULL,
  `PraiseUserId` int(6) NOT NULL,
  `PraiseUserType` smallint(1) NOT NULL DEFAULT '0' COMMENT '默认是学生',
  PRIMARY KEY (`commentId`,`PraiseUserId`,`PraiseUserType`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of praise_comment
-- ----------------------------
INSERT INTO `praise_comment` VALUES ('1', '2', '0');
INSERT INTO `praise_comment` VALUES ('1', '3', '0');
INSERT INTO `praise_comment` VALUES ('1', '4', '0');
INSERT INTO `praise_comment` VALUES ('1', '6', '0');
INSERT INTO `praise_comment` VALUES ('2', '2', '0');
INSERT INTO `praise_comment` VALUES ('2', '4', '0');
INSERT INTO `praise_comment` VALUES ('3', '2', '0');
INSERT INTO `praise_comment` VALUES ('3', '4', '0');
INSERT INTO `praise_comment` VALUES ('4', '2', '0');
INSERT INTO `praise_comment` VALUES ('5', '2', '0');
INSERT INTO `praise_comment` VALUES ('5', '4', '0');
INSERT INTO `praise_comment` VALUES ('6', '2', '0');
INSERT INTO `praise_comment` VALUES ('7', '2', '0');
INSERT INTO `praise_comment` VALUES ('9', '2', '0');
INSERT INTO `praise_comment` VALUES ('10', '2', '0');
INSERT INTO `praise_comment` VALUES ('12', '2', '0');
INSERT INTO `praise_comment` VALUES ('12', '4', '0');
INSERT INTO `praise_comment` VALUES ('13', '2', '0');
INSERT INTO `praise_comment` VALUES ('13', '4', '0');
INSERT INTO `praise_comment` VALUES ('14', '2', '0');
INSERT INTO `praise_comment` VALUES ('14', '4', '0');
INSERT INTO `praise_comment` VALUES ('15', '2', '0');
INSERT INTO `praise_comment` VALUES ('16', '2', '0');
INSERT INTO `praise_comment` VALUES ('17', '2', '0');
INSERT INTO `praise_comment` VALUES ('18', '2', '0');
INSERT INTO `praise_comment` VALUES ('19', '4', '0');
INSERT INTO `praise_comment` VALUES ('21', '2', '0');
INSERT INTO `praise_comment` VALUES ('22', '2', '0');
INSERT INTO `praise_comment` VALUES ('23', '2', '0');
INSERT INTO `praise_comment` VALUES ('24', '1', '1');
INSERT INTO `praise_comment` VALUES ('24', '2', '0');
INSERT INTO `praise_comment` VALUES ('27', '2', '0');
INSERT INTO `praise_comment` VALUES ('28', '2', '0');
INSERT INTO `praise_comment` VALUES ('29', '2', '0');
INSERT INTO `praise_comment` VALUES ('30', '2', '0');
INSERT INTO `praise_comment` VALUES ('33', '2', '0');
INSERT INTO `praise_comment` VALUES ('34', '2', '0');
INSERT INTO `praise_comment` VALUES ('35', '2', '0');

-- ----------------------------
-- Table structure for `schorcom`
-- ----------------------------
DROP TABLE IF EXISTS `schorcom`;
CREATE TABLE `schorcom` (
  `comName` varchar(30) NOT NULL,
  `comId` int(11) NOT NULL AUTO_INCREMENT,
  `type` varchar(20) NOT NULL,
  `comBackground` varchar(50) NOT NULL,
  `comLogo` varchar(50) NOT NULL,
  `comPic` varchar(50) NOT NULL,
  `comIntroduce` varchar(300) NOT NULL,
  PRIMARY KEY (`comId`),
  UNIQUE KEY `comName` (`comName`)
) ENGINE=InnoDB AUTO_INCREMENT=22 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of schorcom
-- ----------------------------
INSERT INTO `schorcom` VALUES ('某野鸡大学1', '1', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', '某野鸡大学1创办于1898年，初名京师大学堂，是中国第一所国立综合性大学，也是当时中国最高教育行政机关。辛亥革命后，于1912年改为现名。 学校为教育部直属全国重点大学，国家“211工程”、“985工程”建设大学、C9联盟，以及东亚研究型大学协会、国际研究型大学联盟、环太平洋大学联盟、东亚四大学论坛的重要成员。');
INSERT INTO `schorcom` VALUES ('某野鸡大学2', '2', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', '某野鸡大学2是一所电子信息特色突出，经管学科优势明显，工、理、经、管、文、法、艺等多学科相互渗透的教学研究型大学。学校建有下沙、文一、东岳、下沙东及信息工程学院临安新校区共五个校区，占地面积2500余亩，现有普通全日制在校生28000余人，教职员工2200余人。');
INSERT INTO `schorcom` VALUES ('某野鸡公司3', '3', 'company', 'daneiback.gif', 'comLogo.png', 'comLogo.png', '时代科技集团有限公司【美股交易代码:TEDU】(简称集团)成立于2002年9月。2014年4月3日成功在美国纳斯达克上市,融资1亿3千万美元。');
INSERT INTO `schorcom` VALUES ('某野鸡大学4', '4', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', '大学（英文名：a University），简称，诞生于1911年，因坐落于北京西北郊的园而得名。初称“学堂”，是清政府设立的留美预备学校；翌年更名为“学校”');
INSERT INTO `schorcom` VALUES ('某野鸡公司5', '5', 'company', 'beijingdaxueback.jpg', 'comLogo.png', 'comLogo.png', '有限公司（英文名： University），简称，诞生于1911年，因坐落于北京西北郊的园而得名。初称“学堂”，是清政府设立的留美预备学校；翌年更名为“学校”');
INSERT INTO `schorcom` VALUES ('某野鸡大学6', '6', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', '大学（英文名 University），简称，诞生于191年，因坐落于北京西北郊的而得名。初称“学堂”，是政府设立的留美预备学校；翌年更名为“学校”');
INSERT INTO `schorcom` VALUES ('某野鸡大学7', '7', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡大学8', '8', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡大学9', '9', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡大学10', '10', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡公司11', '11', 'company', 'beijingdaxueback.jpg', 'comLogo.png', 'comLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡公司12', '12', 'company', 'beijingdaxueback.jpg', 'comLogo.png', 'comLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡公司13', '13', 'company', 'beijingdaxueback.jpg', 'comLogo.png', 'comLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡大学14', '14', 'school', 'beijingdaxueback.jpg', 'schLogo.png', 'schLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('某野鸡公司15', '15', 'company', 'beijingdaxueback.jpg', 'comLogo.png', 'comLogo.png', 'seugv9aeufhcnaeursgvrfhhhhhhhhhhhhhhhhaseu9fe7cnwxgwh08w4rudfvhesrgu8ufgggggggggggggggsrgergu9fhe89weruhgr');
INSERT INTO `schorcom` VALUES ('野鸡大学之王', '21', 'school', 'collection-top.jpg', '6F31DFEDE0DFF8348D3CF978ED909CA4.png', '49E89BEA3F1B3F1AC788F5F94C4A457F.png', '某野鸡大学1创办于1898年，初名京师大学堂，是中国第一所国立综合性大学，也是当时中国最高教育行政机关。辛亥革命后，于1912年改为现名。 学校为教育部直属全国重点大学，国家“211工程”、“985工程”建设大学、C9联盟，以及东亚研究型大学协会、国际研究型大学联盟、环太平洋大学联盟、东亚四大学论坛的重要成员。');

-- ----------------------------
-- Table structure for `smallproblem`
-- ----------------------------
DROP TABLE IF EXISTS `smallproblem`;
CREATE TABLE `smallproblem` (
  `smallProblemId` int(5) NOT NULL AUTO_INCREMENT,
  `belongTestProblemId` int(5) NOT NULL,
  `smallProblemType` smallint(1) NOT NULL COMMENT '小题类型',
  `smallProblemScore` smallint(2) NOT NULL COMMENT '小题分值',
  `smallProblemADetail` varchar(80) DEFAULT NULL,
  `smallProblemBDetail` varchar(80) DEFAULT NULL,
  `smallProblemCDetail` varchar(80) DEFAULT NULL,
  `smallProblemDDetail` varchar(80) DEFAULT NULL,
  `smallProblemTip` varchar(20) DEFAULT NULL,
  `smallProblemTrueAnswer` varchar(50) NOT NULL,
  `smallProblemTitle` varchar(80) NOT NULL,
  PRIMARY KEY (`smallProblemId`)
) ENGINE=InnoDB AUTO_INCREMENT=27 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of smallproblem
-- ----------------------------
INSERT INTO `smallproblem` VALUES ('1', '1', '0', '10', '奥巴马', '习近平', '卢本伟', '黑寡妇', null, 'B', '中国主席是谁');
INSERT INTO `smallproblem` VALUES ('2', '1', '0', '10', '杭州电子科技大学', '清华大学', '北京大学', '麻省理工大学', null, 'A', '以下最牛逼的学校是哪个');
INSERT INTO `smallproblem` VALUES ('3', '1', '0', '10', '世界上最好的大学', '世界上最垃圾的大学', '世界上环境最好的大学', '世界上食堂最好的大学', null, 'B', '以下关于野鸡大学说法正确的是');
INSERT INTO `smallproblem` VALUES ('4', '1', '0', '10', '栈', '队列', '数组', '单向链表', '', 'A', '符合先进后出原则的是');
INSERT INTO `smallproblem` VALUES ('5', '1', '0', '10', '张杰', '周杰伦', '卢本伟', '许嵩', null, 'D', '《山水之间》是谁的作品？');
INSERT INTO `smallproblem` VALUES ('6', '2', '1', '5', null, null, null, null, null, 'false', '世界上最遥远的距离是没网');
INSERT INTO `smallproblem` VALUES ('7', '2', '1', '5', null, null, null, null, null, 'true', '不经历风雨怎能见彩虹');
INSERT INTO `smallproblem` VALUES ('8', '2', '1', '5', null, null, null, null, null, 'false', '大学很好混，很轻松就能毕业的');
INSERT INTO `smallproblem` VALUES ('9', '3', '2', '7', null, null, null, null, 'number', '8', '3+5=_');
INSERT INTO `smallproblem` VALUES ('10', '3', '2', '7', null, null, null, null, 'profeword', '虚拟现实', 'vr中文名字是_');
INSERT INTO `smallproblem` VALUES ('11', '3', '2', '7', null, null, null, null, 'number', '0', '300/35*0=_');
INSERT INTO `smallproblem` VALUES ('12', '3', '2', '7', null, null, null, null, 'number', '-1', '3-4=_');
INSERT INTO `smallproblem` VALUES ('13', '3', '2', '7', null, null, null, null, 'profeword', 'artificial teligence', '人工智能翻译为_');
INSERT INTO `smallproblem` VALUES ('14', '4', '0', '30', '2', '3', '4', '5', 'null', 'A', '1+1=?');
INSERT INTO `smallproblem` VALUES ('15', '4', '0', '30', 'rng', 'ig', 'skt', 'edg', 'null', 'B', 's8lol夺冠队伍是哪支?');
INSERT INTO `smallproblem` VALUES ('16', '5', '1', '10', 'null', 'null', 'null', 'null', 'null', 'true', '杭电是世界上最厉害的大学');
INSERT INTO `smallproblem` VALUES ('17', '5', '1', '10', 'null', 'null', 'null', 'null', 'null', 'true', '杭电计算机学院最牛逼');
INSERT INTO `smallproblem` VALUES ('18', '6', '2', '10', 'null', 'null', 'null', 'null', 'number', '7', '3+4=_');
INSERT INTO `smallproblem` VALUES ('19', '6', '2', '10', 'null', 'null', 'null', 'null', 'profeword', '阴阳割昏晓', '造化钟神秀,_');
INSERT INTO `smallproblem` VALUES ('20', '7', '0', '30', '2', '3', '4', '5', 'null', 'A', '1+1=?');
INSERT INTO `smallproblem` VALUES ('21', '8', '1', '10', 'null', 'null', 'null', 'null', 'null', 'false', '美国现任总统是奥巴马');
INSERT INTO `smallproblem` VALUES ('22', '8', '1', '10', 'null', 'null', 'null', 'null', 'null', 'true', '抗美援朝发生在1953年');
INSERT INTO `smallproblem` VALUES ('23', '9', '0', '30', 'A选项内容1', 'B选项内容1', 'C选项内容1', 'D选项内容1', 'null', 'D', '选择题题目1');
INSERT INTO `smallproblem` VALUES ('24', '9', '0', '30', 'A选项内容2', 'B选项内容2', 'C选项内容2', 'D选项内容2', 'null', 'B', '选择题题目2');
INSERT INTO `smallproblem` VALUES ('25', '10', '1', '20', 'null', 'null', 'null', 'null', 'null', 'true', '判断题题目1');
INSERT INTO `smallproblem` VALUES ('26', '10', '1', '20', 'null', 'null', 'null', 'null', 'null', 'false', '判断题题目2');

-- ----------------------------
-- Table structure for `source`
-- ----------------------------
DROP TABLE IF EXISTS `source`;
CREATE TABLE `source` (
  `sourceId` int(11) NOT NULL AUTO_INCREMENT,
  `belongUtilId` int(8) NOT NULL,
  `sourceSrc` varchar(50) NOT NULL DEFAULT '',
  `sourceType` varchar(12) NOT NULL DEFAULT '',
  `poster` varchar(50) NOT NULL DEFAULT 'defaultVideoPoster.jpg',
  PRIMARY KEY (`sourceId`)
) ENGINE=InnoDB AUTO_INCREMENT=135 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of source
-- ----------------------------
INSERT INTO `source` VALUES ('1', '1', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('2', '2', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('3', '2', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('4', '3', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('5', '4', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('6', '4', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('7', '5', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('8', '6', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('9', '6', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('10', '7', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('11', '8', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('12', '8', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('13', '9', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('14', '10', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('15', '11', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('16', '12', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('17', '12', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('18', '13', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('19', '14', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('20', '14', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('21', '15', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('22', '16', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('23', '16', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('24', '17', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('25', '18', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('26', '18', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('27', '19', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('28', '19', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('29', '20', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('30', '21', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('31', '21', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('32', '22', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('33', '23', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('34', '23', '222.docx', 'text', '-');
INSERT INTO `source` VALUES ('35', '24', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('36', '25', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('37', '26', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('38', '26', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('39', '27', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('40', '28', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('41', '28', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('42', '29', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('43', '30', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('44', '30', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('45', '31', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('46', '32', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('47', '32', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('48', '33', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('49', '33', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('50', '34', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('51', '34', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('52', '35', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('53', '35', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('54', '36', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('55', '36', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('56', '37', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('57', '38', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('58', '39', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('59', '40', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('60', '41', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('61', '42', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('62', '42', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('63', '43', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('64', '44', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('65', '45', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('66', '45', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('67', '46', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('68', '47', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('69', '48', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('70', '49', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('71', '50', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('72', '51', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('73', '52', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('74', '53', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('75', '53', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('76', '54', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('77', '55', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('78', '56', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('79', '56', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('80', '57', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('81', '57', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('82', '58', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('83', '59', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('84', '60', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('85', '61', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('86', '61', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('87', '62', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('88', '63', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('89', '64', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('90', '65', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('91', '66', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('92', '67', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('93', '68', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('94', '69', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('95', '70', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('96', '71', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('97', '72', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('98', '73', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('99', '73', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('100', '74', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('101', '75', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('102', '76', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('103', '77', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('104', '78', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('105', '79', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('106', '80', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('107', '81', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('108', '82', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('109', '82', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('110', '83', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('111', '84', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('112', '85', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('113', '86', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('114', '87', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('115', '88', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('116', '89', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('117', '90', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('118', '91', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('119', '92', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('120', '93', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('121', '94', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('122', '95', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('123', '96', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('124', '97', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('125', '98', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('126', '99', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('127', '100', 'mov_bbb.mp4', 'video', 'defaultVideoPoster.jpg');
INSERT INTO `source` VALUES ('128', '100', 'text1.txt', 'text', '-');
INSERT INTO `source` VALUES ('130', '25', 'Uploadvideo10.mp4', 'video', 'Uploadt01cd9b7a8d861a0f8d.jpg');
INSERT INTO `source` VALUES ('131', '31', 'Upload云智教育项目详细设计文档.wps', 'text', '-');
INSERT INTO `source` VALUES ('132', '101', 'Uploadvideo14.mp4', 'video', 'Uploadt01cd9b7a8d861a0f8d4.jpg');
INSERT INTO `source` VALUES ('133', '101', 'Upload404Error.txt', 'text', '-');
INSERT INTO `source` VALUES ('134', '101', 'Uploadcourse.txt', 'text', '-');

-- ----------------------------
-- Table structure for `student`
-- ----------------------------
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
  `studentId` int(11) NOT NULL AUTO_INCREMENT,
  `studentAccount` varchar(30) NOT NULL COMMENT '学员账号',
  `studentName` varchar(30) NOT NULL COMMENT '姓名',
  `studentPassword` varchar(30) NOT NULL COMMENT '密码',
  `type` varchar(10) NOT NULL DEFAULT '学生' COMMENT '类型',
  `studentPic` varchar(100) NOT NULL DEFAULT 'noavatar_big.jpg' COMMENT '头像',
  `studentIntroduce` varchar(300) NOT NULL DEFAULT '我有点懒 什么都没留下~' COMMENT '自我介绍',
  `commentNum` int(3) NOT NULL DEFAULT '0',
  `belongClass` varchar(10) NOT NULL DEFAULT '',
  `belongSchId` varchar(12) NOT NULL DEFAULT '',
  `createDate` varchar(20) NOT NULL COMMENT '账号创建日期',
  `studentEmail` varchar(30) NOT NULL,
  `belongSchName` varchar(80) NOT NULL DEFAULT '',
  `studentSex` varchar(6) NOT NULL DEFAULT '女',
  `isBelongSch` int(1) NOT NULL DEFAULT '0',
  `breakTopicRuleNum` int(2) NOT NULL DEFAULT '0' COMMENT '违规发表话题次数',
  `breakComRuleNum` int(2) NOT NULL DEFAULT '0' COMMENT '违规发表评论次数',
  PRIMARY KEY (`studentId`),
  UNIQUE KEY `studentAccount` (`studentAccount`),
  UNIQUE KEY `studentAccount_2` (`studentAccount`),
  UNIQUE KEY `studentAccount_3` (`studentAccount`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of student
-- ----------------------------
INSERT INTO `student` VALUES ('1', '15051527', '王丽', '123456', '学生', 'Uploadt01cd9b7a8d861a0f8d.jpg', '我是来自某野鸡大学2大三计算机学院的一名高材生。我十分厉害', '3', '15052315', '2', '2018-03-01', '152684@qq.com', '某野鸡大学2', '女', '1', '0', '0');
INSERT INTO `student` VALUES ('2', '1536801671xqq1', '张三', '666666', '学生', 'Uploadt01cd9b7a8d861a0f8d3.jpg', '我是一名优秀的学生', '0', '15052314', '1', '2018-04-01', '12580@163.com', '某野鸡大学1', '男', '1', '0', '5');
INSERT INTO `student` VALUES ('3', '15052318xqq', '鹿晗', '111111', '学生', '08CB4D1EC4928B8C5174B6083B49CDF8.jpg', '我是来自某野鸡大学9的一名X学院的学生', '0', '15021414', '9', '2018-04-01', '94859@qq.com', '某野鸡大学9', '男', '1', '0', '0');
INSERT INTO `student` VALUES ('4', '1536801671', 'helloword', '233333', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '14258@qq.com', '', '男', '0', '0', '0');
INSERT INTO `student` VALUES ('5', '9999999', 'uzi', '123456789', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '484@qq.com', '', '女', '0', '0', '0');
INSERT INTO `student` VALUES ('6', '999', 'cnm', '111', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '333@qq.com', '', '女', '0', '0', '0');
INSERT INTO `student` VALUES ('7', '6789', 'sb', '12345', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '123@qq.com', '', '女', '0', '0', '0');
INSERT INTO `student` VALUES ('8', '258', 'sb11', '123', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '123@qq.com', '', '女', '0', '0', '0');
INSERT INTO `student` VALUES ('9', '77852', '123', '123', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '8888@163.com', '', '女', '0', '0', '0');
INSERT INTO `student` VALUES ('10', '125869', 'zhang', '222', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '123@qq.com', '', '女', '0', '0', '0');
INSERT INTO `student` VALUES ('11', '9999', '卢姥爷', '123456', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2018-04-01', '1549884@qq.com', '', '男', '0', '0', '0');
INSERT INTO `student` VALUES ('13', '18268022593', 'CTRP', '666666', '学生', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '', '', '2019-05-31', 'jntm@cxk.com', '', '女', '0', '0', '0');

-- ----------------------------
-- Table structure for `stu_course`
-- ----------------------------
DROP TABLE IF EXISTS `stu_course`;
CREATE TABLE `stu_course` (
  `studentId` int(5) NOT NULL,
  `courseId` int(6) NOT NULL,
  `chooseTime` date NOT NULL,
  `examScore` double(4,0) NOT NULL DEFAULT '-1' COMMENT '-1默认未做，-2默认已做，但老师未批改',
  `AllScore` double(4,1) NOT NULL DEFAULT '-1.0',
  `commentScore` double(4,1) NOT NULL DEFAULT '-1.0',
  `testScore` double(4,1) NOT NULL DEFAULT '-1.0',
  `oneAnswerSrc` varchar(80) DEFAULT NULL,
  `twoAnswerSrc` varchar(80) DEFAULT NULL,
  `threeAnswerSrc` varchar(80) DEFAULT NULL,
  `fourAnswerSrc` varchar(80) DEFAULT NULL,
  `fiveAnswerSrc` varchar(80) DEFAULT NULL,
  PRIMARY KEY (`studentId`,`courseId`),
  KEY `FK_ID6` (`courseId`),
  CONSTRAINT `FK_ID5` FOREIGN KEY (`studentId`) REFERENCES `student` (`studentId`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK_ID6` FOREIGN KEY (`courseId`) REFERENCES `course` (`courseId`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of stu_course
-- ----------------------------
INSERT INTO `stu_course` VALUES ('1', '1', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '2', '2019-04-09', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '3', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '4', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '5', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '7', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '8', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '11', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '16', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '21', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '103', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('1', '114', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('2', '1', '2019-05-22', '41', '32.0', '78.0', '0.0', 'Uploadt01cd9b7a8d861a0f8d0.jpg', 'Uploade677-hhvciiv74657113.jpg', 'Uploadu=9200871,1663220040&fm=27&gp=062.jpg', 'Uploadt01928551cfd344bc44webp4.jpg', 'Uploadhuangzu8.jpeg');
INSERT INTO `stu_course` VALUES ('2', '2', '2019-04-01', '97', '74.4', '78.0', '50.0', 'Uploadhuangzu5.jpeg', 'Uploadu=9200871,1663220040&fm=27&gp=06.jpg', 'UploadZOCYhhUdUwereQBLzhpLdEtZOaUqWgW2EWQoSDRDheWr015090071071708.jpg', 'Uploade677-hhvciiv74657116.jpg', 'Uploadt01928551cfd344bc44webp7.jpg');
INSERT INTO `stu_course` VALUES ('2', '3', '2019-04-10', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('3', '1', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('4', '1', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('4', '2', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('4', '3', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('4', '4', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('4', '9', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('4', '11', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('8', '1', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('8', '3', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('8', '5', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('11', '1', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('11', '2', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('11', '3', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('11', '5', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('11', '17', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);
INSERT INTO `stu_course` VALUES ('11', '21', '2019-04-01', '-1', '-1.0', '-1.0', '-1.0', null, null, null, null, null);

-- ----------------------------
-- Table structure for `teacher`
-- ----------------------------
DROP TABLE IF EXISTS `teacher`;
CREATE TABLE `teacher` (
  `teacherAccount` varchar(30) NOT NULL COMMENT '学员账号',
  `teacherId` int(11) NOT NULL AUTO_INCREMENT,
  `teacherName` varchar(30) NOT NULL COMMENT '姓名',
  `teacherPassword` varchar(30) NOT NULL COMMENT '密码',
  `type` varchar(10) NOT NULL COMMENT '类型',
  `teacherPic` varchar(100) NOT NULL DEFAULT 'noavatar_big.jpg' COMMENT '头像',
  `teacherIntroduce` varchar(300) NOT NULL DEFAULT '我有点懒 什么都没留下~' COMMENT '自我介绍',
  `commentNum` int(3) NOT NULL DEFAULT '0',
  `belongSchId` varchar(12) NOT NULL DEFAULT '',
  `createDate` varchar(20) NOT NULL COMMENT '账号创建日期',
  `Email` varchar(30) NOT NULL,
  `belongSchName` varchar(80) NOT NULL DEFAULT '',
  `teacherSex` varchar(6) NOT NULL DEFAULT '女',
  `teacherLevel` varchar(30) NOT NULL DEFAULT '老师',
  `teacherPhone` varchar(11) NOT NULL DEFAULT '18685025897',
  PRIMARY KEY (`teacherId`),
  UNIQUE KEY `studentAccount` (`teacherAccount`),
  UNIQUE KEY `studentAccount_2` (`teacherAccount`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of teacher
-- ----------------------------
INSERT INTO `teacher` VALUES ('JS15051527', '1', '杨幂', '123456', '老师', 'Uploadt01928551cfd344bc44webp.jpg', '我是来自某野鸡大学2大三计算机学院的一名教授。我十分厉害', '3', '2', '2018年3月31日', '152684@qq.com', '某野鸡大学2', '女', '教授', '18685025897');
INSERT INTO `teacher` VALUES ('JS1536801671xqq1', '2', '卢本伟', '666666', '老师', 'UploadZOCYhhUdUwereQBLzhpLdEtZOaUqWgW2EWQoSDRDheWr01509007107170.jpg', '我是一名优秀的教师', '0', '1', '2018年4月1日', '12580@163.com', '某野鸡大学1', '男', '老师', '18385025897');
INSERT INTO `teacher` VALUES ('JS15052318xqq', '3', '赵信', '111111', '老师', '2F2B3E11578DC027FBC1B022EAA126C8.jpg', '我是来自某野鸡大学9的一名X学院的大师', '0', '9', '2018年4月2日', '94859@qq.com', '某野鸡大学9', '男', '副教授', '18655025893');
INSERT INTO `teacher` VALUES ('JS1536801671', '4', '高宏杰', '233333', '老师', 'noavatar_big.jpg', '我有点懒 什么都没留下~', '0', '4', '2018年4月1日', '14258@qq.com', '某野鸡大学4', '女', '教授', '13685025475');

-- ----------------------------
-- Table structure for `teach_course`
-- ----------------------------
DROP TABLE IF EXISTS `teach_course`;
CREATE TABLE `teach_course` (
  `teacherId` int(5) NOT NULL,
  `courseId` int(6) NOT NULL,
  `examAgree` tinyint(2) NOT NULL DEFAULT '-1' COMMENT '-1表示未看，0表示不同意，1表示同意',
  `examArgumentSuplement` varchar(200) NOT NULL DEFAULT 'not read',
  PRIMARY KEY (`teacherId`,`courseId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of teach_course
-- ----------------------------
INSERT INTO `teach_course` VALUES ('1', '2', '1', '同意发布');
INSERT INTO `teach_course` VALUES ('1', '3', '-1', 'not read');
INSERT INTO `teach_course` VALUES ('2', '1', '1', 'pass');
INSERT INTO `teach_course` VALUES ('2', '2', '1', 'pass');
INSERT INTO `teach_course` VALUES ('2', '3', '-1', 'not read');
INSERT INTO `teach_course` VALUES ('2', '150', '-1', 'not read');
INSERT INTO `teach_course` VALUES ('4', '1', '1', '同意发布');
INSERT INTO `teach_course` VALUES ('4', '3', '-1', 'not read');

-- ----------------------------
-- Table structure for `test`
-- ----------------------------
DROP TABLE IF EXISTS `test`;
CREATE TABLE `test` (
  `testId` int(7) NOT NULL AUTO_INCREMENT,
  `testName` varchar(30) NOT NULL,
  `testStartTime` varchar(30) NOT NULL,
  `testEndTime` date NOT NULL,
  `testAllMark` int(4) NOT NULL DEFAULT '100',
  `submitCount` int(2) NOT NULL DEFAULT '3',
  `testType` varchar(20) NOT NULL DEFAULT '单元测验',
  `limitTime` varchar(3) NOT NULL DEFAULT '',
  `testIntro` varchar(150) NOT NULL DEFAULT '',
  `belongChapterId` int(5) NOT NULL,
  `publicTeacherId` int(6) NOT NULL,
  `testState` smallint(1) NOT NULL DEFAULT '-1' COMMENT '默认为未正式发布',
  PRIMARY KEY (`testId`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of test
-- ----------------------------
INSERT INTO `test` VALUES ('1', '第五单元测验', '2018年4月1日', '2019-04-21', '100', '3', '单元测验', '25', '本次作业由10道选择题和10道判断题以及5道填空题组成    总分共 14', '14', '1', '1');
INSERT INTO `test` VALUES ('2', '第6单元测验', '2019年4月19日', '2019-04-26', '100', '3', '课后作业', '20', 'auwhguhawegoehrghoera', '13', '1', '1');
INSERT INTO `test` VALUES ('3', '第一单元测验', '2019年4月21日', '2019-04-26', '50', '3', '单元测验', '10', 'ahwoighowrighrihho', '9', '1', '1');
INSERT INTO `test` VALUES ('4', '第三单元测验', '2019-04-21', '2019-04-25', '100', '3', '单元测验', '', '请同学们按时完成第二单元测验', '14', '1', '-1');
INSERT INTO `test` VALUES ('5', '单元测验', '2019-05-01', '2019-05-06', '100', '3', '单元测验', '', 'ahwoighaiwjfwejapogew', '11', '1', '1');

-- ----------------------------
-- Table structure for `testproblem`
-- ----------------------------
DROP TABLE IF EXISTS `testproblem`;
CREATE TABLE `testproblem` (
  `testProblemId` int(5) NOT NULL AUTO_INCREMENT,
  `belongTestId` int(5) NOT NULL,
  `testProblemType` smallint(1) NOT NULL,
  `testProblemOrder` int(2) NOT NULL,
  `testProblemTitle` varchar(50) NOT NULL COMMENT '大题标题',
  PRIMARY KEY (`testProblemId`),
  KEY `FK_ID1` (`belongTestId`),
  CONSTRAINT `FK_ID1` FOREIGN KEY (`belongTestId`) REFERENCES `test` (`testId`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of testproblem
-- ----------------------------
INSERT INTO `testproblem` VALUES ('1', '1', '0', '1', '选择题，共5道小题，每道小题10分，总共5分');
INSERT INTO `testproblem` VALUES ('2', '1', '1', '2', '判断题，共3道小题，每题5分，总分15分');
INSERT INTO `testproblem` VALUES ('3', '1', '2', '3', '填空题，共5道小题，每题7分，总分35分');
INSERT INTO `testproblem` VALUES ('4', '2', '0', '1', '选择题，共2道小题,每道题30分，总共60分');
INSERT INTO `testproblem` VALUES ('5', '2', '1', '2', '判断题，共2道小题,每道题10分，总共20分');
INSERT INTO `testproblem` VALUES ('6', '2', '2', '3', '填空题，共2道小题,每道题10分，总共20分');
INSERT INTO `testproblem` VALUES ('7', '3', '0', '1', '选择题，共1道小题,每道题30分，总共30分');
INSERT INTO `testproblem` VALUES ('8', '3', '1', '2', '判断题，共2道小题,每道题10分，总共20分');
INSERT INTO `testproblem` VALUES ('9', '5', '0', '1', '选择题，共2道小题,每道题30分，总共60分');
INSERT INTO `testproblem` VALUES ('10', '5', '1', '2', '判断题，共2道小题,每道题20分，总共40分');

-- ----------------------------
-- Table structure for `test_score`
-- ----------------------------
DROP TABLE IF EXISTS `test_score`;
CREATE TABLE `test_score` (
  `scoreId` int(11) NOT NULL AUTO_INCREMENT,
  `studentId` int(8) NOT NULL,
  `score` float(3,0) NOT NULL,
  `testId` int(8) NOT NULL,
  `alreadyCount` int(2) NOT NULL DEFAULT '0',
  PRIMARY KEY (`scoreId`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of test_score
-- ----------------------------
INSERT INTO `test_score` VALUES ('1', '2', '63', '2', '1');
INSERT INTO `test_score` VALUES ('2', '2', '20', '1', '3');
INSERT INTO `test_score` VALUES ('3', '1', '80', '5', '3');
INSERT INTO `test_score` VALUES ('5', '2', '100', '5', '3');

-- ----------------------------
-- Table structure for `topic`
-- ----------------------------
DROP TABLE IF EXISTS `topic`;
CREATE TABLE `topic` (
  `topicId` int(6) NOT NULL AUTO_INCREMENT,
  `belongCourseId` int(6) NOT NULL,
  `topicDetail` varchar(150) NOT NULL,
  `topicTime` varchar(20) NOT NULL,
  `topicWritterId` int(6) NOT NULL,
  `topicWritterType` smallint(1) NOT NULL,
  `reportNum` int(3) NOT NULL DEFAULT '0' COMMENT '举报次数，超过5次提交给后台管理员审核是否屏蔽',
  `isForbiden` smallint(1) NOT NULL DEFAULT '0' COMMENT '是否对这个话题进行屏蔽,默认为不屏蔽',
  PRIMARY KEY (`topicId`)
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of topic
-- ----------------------------
INSERT INTO `topic` VALUES ('1', '1', '怎样做一个年少有为的当代青年？', '2019-02-02 14:24:37', '1', '1', '2', '0');
INSERT INTO `topic` VALUES ('2', '1', 'why do you study this course?', '2019-02-01 06:27:05', '2', '1', '0', '0');
INSERT INTO `topic` VALUES ('3', '1', '到底要怎么才能上王者？', '2019-02-03 13:24:35', '2', '0', '4', '1');
INSERT INTO `topic` VALUES ('4', '1', '孙悟空怎么对线吸血鬼？\n吸血鬼真的恶心，怎么打', '2019-02-03 14:03:26', '2', '0', '0', '0');
INSERT INTO `topic` VALUES ('5', '1', '11111111111111111111111111111111111111111111111111111111111111111111111111111111111111', '2019-02-03 14:04:46', '2', '0', '6', '0');
INSERT INTO `topic` VALUES ('7', '1', '？？？', '2019-02-03 23:52:54', '2', '0', '3', '1');
INSERT INTO `topic` VALUES ('8', '1', '计算机组成原理这门课的重要性？', '2019-02-05 18:09:59', '2', '0', '0', '0');
INSERT INTO `topic` VALUES ('9', '3', '这门课平时分多少？', '2019-02-05 19:32:40', '2', '0', '0', '0');
INSERT INTO `topic` VALUES ('10', '2', '呵呵', '2019-04-05 15:57:27', '2', '0', '1', '0');
INSERT INTO `topic` VALUES ('11', '2', '嘤嘤嘤', '2019-04-05 16:04:08', '2', '0', '12', '0');
INSERT INTO `topic` VALUES ('12', '2', 'hello', '2019-04-06 19:06:42', '1', '0', '0', '0');
INSERT INTO `topic` VALUES ('13', '3', '？？？', '2019-04-14 15:52:03', '1', '1', '1', '0');
INSERT INTO `topic` VALUES ('14', '1', 'aweigjrig', '2019-05-01 11:16:53', '2', '0', '0', '0');
INSERT INTO `topic` VALUES ('15', '1', 'hello everybody', '2019-05-04 15:04:07', '1', '0', '0', '0');

-- ----------------------------
-- Table structure for `training`
-- ----------------------------
DROP TABLE IF EXISTS `training`;
CREATE TABLE `training` (
  `id` int(5) NOT NULL AUTO_INCREMENT,
  `course` int(6) NOT NULL,
  `name` varchar(30) NOT NULL DEFAULT '',
  `content` varchar(300) NOT NULL DEFAULT '',
  `releaseDate` varchar(30) NOT NULL DEFAULT '',
  `deadline` varchar(30) NOT NULL DEFAULT '',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of training
-- ----------------------------
INSERT INTO `training` VALUES ('1', '8', '文件系统实现', '一个月之内完成linux操作系统中的文件系统部分', '2018-05-09', '2018-05-24');
INSERT INTO `training` VALUES ('2', '8', '文件系统实现', '一个月之内完成linux操作系统中的文件系统部分', '2018-05-09', '2018-05-24');

-- ----------------------------
-- Table structure for `util`
-- ----------------------------
DROP TABLE IF EXISTS `util`;
CREATE TABLE `util` (
  `utilId` int(11) NOT NULL AUTO_INCREMENT,
  `utilTitle` varchar(20) NOT NULL,
  `utilName` varchar(30) NOT NULL,
  `belongChapterId` int(11) NOT NULL,
  `utilOrder` int(3) NOT NULL,
  PRIMARY KEY (`utilId`)
) ENGINE=InnoDB AUTO_INCREMENT=107 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of util
-- ----------------------------
INSERT INTO `util` VALUES ('1', '第一小节', '欢迎来到英雄联盟', '1', '1');
INSERT INTO `util` VALUES ('2', '第二小节', '欢迎来到绝地求生', '1', '2');
INSERT INTO `util` VALUES ('3', '第一小节', '欢迎来到王者荣耀', '2', '1');
INSERT INTO `util` VALUES ('4', '第二小节', '欢迎来到绝地求生', '2', '2');
INSERT INTO `util` VALUES ('5', '第一小节', '欢迎来到王者荣耀', '3', '1');
INSERT INTO `util` VALUES ('6', '第二小节', '欢迎来到英雄联盟', '3', '2');
INSERT INTO `util` VALUES ('7', '第一小节', '欢迎来到英雄联盟', '4', '1');
INSERT INTO `util` VALUES ('8', '第二小节', '欢迎来到绝地求生', '4', '2');
INSERT INTO `util` VALUES ('9', '第一小节', '欢迎来到王者荣耀', '5', '1');
INSERT INTO `util` VALUES ('10', '第二小节', '欢迎来到绝地求生', '5', '2');
INSERT INTO `util` VALUES ('11', '第一小节', '欢迎来到王者荣耀', '6', '1');
INSERT INTO `util` VALUES ('12', '第二小节', '欢迎来到绝地求生', '6', '2');
INSERT INTO `util` VALUES ('13', '第一小节', '欢迎来到英雄联盟', '7', '1');
INSERT INTO `util` VALUES ('14', '第二小节', '欢迎来到绝地求生', '7', '2');
INSERT INTO `util` VALUES ('15', '第一小节', '欢迎来到王者荣耀', '8', '1');
INSERT INTO `util` VALUES ('16', '第二小节', '欢迎来到绝地求生', '8', '2');
INSERT INTO `util` VALUES ('17', '第一小节', '欢迎来到王者荣耀', '9', '1');
INSERT INTO `util` VALUES ('18', '第二小节', '欢迎来到绝地求生', '9', '2');
INSERT INTO `util` VALUES ('19', '第一小节', '欢迎来到英雄联盟', '10', '1');
INSERT INTO `util` VALUES ('20', '第二小节', '欢迎来到绝地求生', '10', '2');
INSERT INTO `util` VALUES ('21', '第一小节', '欢迎来到王者荣耀', '11', '1');
INSERT INTO `util` VALUES ('22', '第二小节', '欢迎来到绝地求生', '11', '2');
INSERT INTO `util` VALUES ('23', '第一小节', '欢迎来到王者荣耀', '12', '1');
INSERT INTO `util` VALUES ('24', '第二小节', '欢迎来到绝地求生', '12', '2');
INSERT INTO `util` VALUES ('25', '第一小节', '欢迎来到英雄联盟', '13', '1');
INSERT INTO `util` VALUES ('26', '第二小节', '欢迎来到绝地求生', '13', '2');
INSERT INTO `util` VALUES ('27', '第一小节', '欢迎来到王者荣耀', '14', '1');
INSERT INTO `util` VALUES ('28', '第二小节', '欢迎来到绝地求生', '14', '2');
INSERT INTO `util` VALUES ('29', '第一小节', '欢迎来到王者荣耀', '15', '1');
INSERT INTO `util` VALUES ('30', '第二小节', '欢迎来到绝地求生', '15', '2');
INSERT INTO `util` VALUES ('31', '第一小节', '欢迎来到英雄联盟', '16', '1');
INSERT INTO `util` VALUES ('32', '第二小节', '欢迎来到绝地求生', '16', '2');
INSERT INTO `util` VALUES ('33', '第一小节', '欢迎来到王者荣耀', '17', '1');
INSERT INTO `util` VALUES ('34', '第二小节', '欢迎来到绝地求生', '17', '2');
INSERT INTO `util` VALUES ('35', '第一小节', '欢迎来到王者荣耀', '18', '1');
INSERT INTO `util` VALUES ('36', '第二小节', '欢迎来到绝地求生', '18', '2');
INSERT INTO `util` VALUES ('37', '第一小节', '欢迎来到英雄联盟', '19', '1');
INSERT INTO `util` VALUES ('38', '第二小节', '欢迎来到绝地求生', '19', '2');
INSERT INTO `util` VALUES ('39', '第一小节', '欢迎来到王者荣耀', '20', '1');
INSERT INTO `util` VALUES ('40', '第二小节', '欢迎来到绝地求生', '20', '2');
INSERT INTO `util` VALUES ('41', '第一小节', '欢迎来到王者荣耀', '21', '1');
INSERT INTO `util` VALUES ('42', '第二小节', '欢迎来到绝地求生', '21', '2');
INSERT INTO `util` VALUES ('43', '第一小节', '欢迎来到英雄联盟', '22', '1');
INSERT INTO `util` VALUES ('44', '第二小节', '欢迎来到绝地求生', '22', '2');
INSERT INTO `util` VALUES ('45', '第一小节', '欢迎来到王者荣耀', '23', '1');
INSERT INTO `util` VALUES ('46', '第二小节', '欢迎来到绝地求生', '23', '2');
INSERT INTO `util` VALUES ('47', '第一小节', '欢迎来到王者荣耀', '24', '1');
INSERT INTO `util` VALUES ('48', '第二小节', '欢迎来到绝地求生', '24', '2');
INSERT INTO `util` VALUES ('49', '第一小节', '欢迎来到英雄联盟', '25', '1');
INSERT INTO `util` VALUES ('50', '第二小节', '欢迎来到绝地求生', '25', '2');
INSERT INTO `util` VALUES ('51', '第一小节', '欢迎来到王者荣耀', '26', '1');
INSERT INTO `util` VALUES ('52', '第二小节', '欢迎来到绝地求生', '26', '2');
INSERT INTO `util` VALUES ('53', '第一小节', '欢迎来到王者荣耀', '27', '1');
INSERT INTO `util` VALUES ('54', '第二小节', '欢迎来到绝地求生', '27', '2');
INSERT INTO `util` VALUES ('55', '第一小节', '欢迎来到英雄联盟', '28', '1');
INSERT INTO `util` VALUES ('56', '第二小节', '欢迎来到绝地求生', '28', '2');
INSERT INTO `util` VALUES ('57', '第一小节', '欢迎来到王者荣耀', '29', '1');
INSERT INTO `util` VALUES ('58', '第二小节', '欢迎来到绝地求生', '29', '2');
INSERT INTO `util` VALUES ('59', '第一小节', '欢迎来到王者荣耀', '30', '1');
INSERT INTO `util` VALUES ('60', '第二小节', '欢迎来到绝地求生', '30', '2');
INSERT INTO `util` VALUES ('61', '第一小节', '欢迎来到英雄联盟', '31', '1');
INSERT INTO `util` VALUES ('62', '第二小节', '欢迎来到绝地求生', '31', '2');
INSERT INTO `util` VALUES ('63', '第一小节', '欢迎来到王者荣耀', '32', '1');
INSERT INTO `util` VALUES ('64', '第二小节', '欢迎来到绝地求生', '32', '2');
INSERT INTO `util` VALUES ('65', '第一小节', '欢迎来到王者荣耀', '33', '1');
INSERT INTO `util` VALUES ('66', '第二小节', '欢迎来到绝地求生', '33', '2');
INSERT INTO `util` VALUES ('67', '第一小节', '欢迎来到英雄联盟', '34', '1');
INSERT INTO `util` VALUES ('68', '第二小节', '欢迎来到绝地求生', '34', '2');
INSERT INTO `util` VALUES ('69', '第一小节', '欢迎来到王者荣耀', '35', '1');
INSERT INTO `util` VALUES ('70', '第二小节', '欢迎来到绝地求生', '35', '2');
INSERT INTO `util` VALUES ('71', '第一小节', '欢迎来到王者荣耀', '36', '1');
INSERT INTO `util` VALUES ('72', '第二小节', '欢迎来到绝地求生', '36', '2');
INSERT INTO `util` VALUES ('73', '第一小节', '欢迎来到英雄联盟', '37', '1');
INSERT INTO `util` VALUES ('74', '第二小节', '欢迎来到绝地求生', '37', '2');
INSERT INTO `util` VALUES ('75', '第一小节', '欢迎来到王者荣耀', '38', '1');
INSERT INTO `util` VALUES ('76', '第二小节', '欢迎来到绝地求生', '38', '2');
INSERT INTO `util` VALUES ('77', '第一小节', '欢迎来到王者荣耀', '39', '1');
INSERT INTO `util` VALUES ('78', '第二小节', '欢迎来到绝地求生', '39', '2');
INSERT INTO `util` VALUES ('79', '第一小节', '欢迎来到英雄联盟', '40', '1');
INSERT INTO `util` VALUES ('80', '第二小节', '欢迎来到绝地求生', '40', '2');
INSERT INTO `util` VALUES ('81', '第一小节', '欢迎来到王者荣耀', '41', '1');
INSERT INTO `util` VALUES ('82', '第二小节', '欢迎来到绝地求生', '42', '2');
INSERT INTO `util` VALUES ('83', '第一小节', '欢迎来到王者荣耀', '43', '1');
INSERT INTO `util` VALUES ('84', '第二小节', '欢迎来到绝地求生', '43', '2');
INSERT INTO `util` VALUES ('85', '第一小节', '欢迎来到英雄联盟', '44', '1');
INSERT INTO `util` VALUES ('86', '第二小节', '欢迎来到绝地求生', '44', '2');
INSERT INTO `util` VALUES ('87', '第一小节', '欢迎来到王者荣耀', '45', '1');
INSERT INTO `util` VALUES ('88', '第二小节', '欢迎来到绝地求生', '45', '2');
INSERT INTO `util` VALUES ('89', '第一小节', '欢迎来到王者荣耀', '46', '1');
INSERT INTO `util` VALUES ('90', '第二小节', '欢迎来到绝地求生', '46', '2');
INSERT INTO `util` VALUES ('91', '第一小节', '欢迎来到英雄联盟', '47', '1');
INSERT INTO `util` VALUES ('92', '第二小节', '欢迎来到绝地求生', '47', '2');
INSERT INTO `util` VALUES ('93', '第一小节', '欢迎来到王者荣耀', '48', '1');
INSERT INTO `util` VALUES ('94', '第二小节', '欢迎来到绝地求生', '48', '2');
INSERT INTO `util` VALUES ('95', '第一小节', '欢迎来到王者荣耀', '49', '1');
INSERT INTO `util` VALUES ('96', '第二小节', '欢迎来到绝地求生', '49', '2');
INSERT INTO `util` VALUES ('97', '第一小节', '欢迎来到英雄联盟', '50', '1');
INSERT INTO `util` VALUES ('98', '第二小节', '欢迎来到绝地求生', '50', '2');
INSERT INTO `util` VALUES ('99', '第一小节', '欢迎来到王者荣耀', '51', '1');
INSERT INTO `util` VALUES ('100', '第二小节', '欢迎来到绝地求生', '51', '2');
INSERT INTO `util` VALUES ('101', '第一小节', '网吧战队', '52', '1');
INSERT INTO `util` VALUES ('102', '第二小节', '加入皇族', '52', '2');
INSERT INTO `util` VALUES ('103', '第三小节', '皇族亚军', '52', '3');
INSERT INTO `util` VALUES ('104', '第一小节', '遇到UU', '53', '1');
INSERT INTO `util` VALUES ('105', '第二小节', '开挂风波', '53', '2');
INSERT INTO `util` VALUES ('106', '第三小节', '给卢姥爷上香', '53', '3');
