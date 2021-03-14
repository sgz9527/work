package com.xqq.service;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xqq.constant.Constants;
import com.xqq.dao.IUserDao;
import com.xqq.dao.ManagerMapper;
import com.xqq.dao.StudentMapper;
import com.xqq.dao.TeacherMapper;
import com.xqq.pojo.Manager;
import com.xqq.pojo.Student;
import com.xqq.pojo.Teacher;
import com.xqq.staticmethod.SignatureUtils;
import com.xqq.staticmethod.faceRecongnize;
import com.xqq.staticmethod.getDate;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;

@Service("LoginService")
public class LoginServiceImpl implements LoginService {
    @Resource(name = "UserDao")
    IUserDao userDao;
    @Resource
    StudentMapper studentDao;
    @Resource(name = "TeacherMapper")
    TeacherMapper teacherDao;
    @Resource
    ManagerMapper managerDao;
    @Override
    public Student LoginStudent(String StudentAccount, String StudentPassword) {
        return userDao.LoginStudent(StudentAccount, StudentPassword);
    }

    @Override
    public Teacher LoginTeacher(String TeacherAccount, String TeacherPassword) {
        return teacherDao.selectByAccountAndPassword(TeacherAccount, TeacherPassword);
    }

    @Override
    public Manager loginAdmin(String account, String password) {
        return managerDao.selectByAccountAndPassword(account, password);
    }

    @Override
    public HashMap<String, Object> faceAdminLogin  (String baseData) throws Exception{
        System.out.println("LoginServiceImpl.faceAdminLogin");
        HashMap<String,Object> map1=new HashMap<>(1);
        String newUrl;
        newUrl=this.generateNewUrl(baseData);
        System.out.println("LoginServiceImpl.faceAdminLogin,result:"+this.sendGet(newUrl));
        map1.put("result","ok");
        return map1;
    }

    @Override
    public Manager faceAdminLoginBaidu(String baseData) {
        String result=faceRecongnize.search(baseData,Constants.ADMIN_GROUP_ID);
        JsonObject jsonObj = null;
            JsonParser jspa=new JsonParser();
            JsonElement jsElement=jspa.parse(result);
            jsonObj = jsElement.getAsJsonObject();
            int errorCode=jsonObj.get("error_code").getAsInt();
            if(errorCode==0) {
                JsonObject resultJson =  jsonObj.get("result").getAsJsonObject();
                JsonArray jsonArr=resultJson.get("user_list").getAsJsonArray();
                String userAccount="null";
                Double score = 0.0;
                for(int i=0;i<jsonArr.size();i++){
                    JsonObject jsob=jsonArr.get(i).getAsJsonObject();
                    Double thisScore=jsob.get("score").getAsDouble();
                    if(thisScore>=Constants.FACE_PASS_MIN_SCORE) {
                        if (thisScore >= score) {
                            score = thisScore;
                            userAccount = jsob.get("user_id").getAsString();
                        }
                    }
                }
                    System.out.println("LoginServiceImpl.faceAdminLoginBaidu,userAccount:" + userAccount + ",similar score:" + score);
                    if (score >= Constants.FACE_PASS_MIN_SCORE) {
                        Manager manager = managerDao.selectByAccount(userAccount);
                        if (manager != null) {
                            return manager;
                        }
                    }
            }
        return null;
    }

    @Override
    public String faceAdd(String baseData,String groupId,String userAccount) {
        return faceRecongnize.add(baseData,groupId,userAccount);
    }

    @Override
    public Student faceLoginStuBaidu(String baseData) {
        String result=faceRecongnize.search(baseData,Constants.STUDENT_GROUP_ID);
        JsonObject jsonObj = null;
        JsonParser jspa=new JsonParser();
        JsonElement jsElement=jspa.parse(result);
        jsonObj = jsElement.getAsJsonObject();
        int errorCode=jsonObj.get("error_code").getAsInt();
        if(errorCode==0) {
            JsonObject resultJson =  jsonObj.get("result").getAsJsonObject();
            JsonArray jsonArr=resultJson.get("user_list").getAsJsonArray();
            String userAccount="null";
            Double score = 0.0;
            for(int i=0;i<jsonArr.size();i++){
                JsonObject jsob=jsonArr.get(i).getAsJsonObject();
                Double thisScore=jsob.get("score").getAsDouble();
                if(thisScore>=Constants.FACE_PASS_MIN_SCORE) {
                    if (thisScore >= score) {
                        userAccount = jsob.get("user_id").getAsString();
                        score = thisScore;
                    }
                }
            }
            System.out.println("LoginServiceImpl.faceLoginStuBaidu,studentAccount:" + userAccount + ",similar score:" + score);
            if (score >= Constants.FACE_PASS_MIN_SCORE) {
                Student student = userDao.selectByAccount(userAccount);
                if (student != null) {
                    return student;
                }
            }
        }
        return null;
    }

    @Override
    public Teacher faceLoginTeachBaidu(String baseData) {
        String result=faceRecongnize.search(baseData,Constants.TEACHER_GROUP_ID);
        JsonObject jsonObj = null;
        JsonParser jspa=new JsonParser();
        JsonElement jsElement=jspa.parse(result);
        jsonObj = jsElement.getAsJsonObject();
        int errorCode=jsonObj.get("error_code").getAsInt();
        if(errorCode==0) {
            JsonObject resultJson =  jsonObj.get("result").getAsJsonObject();
            JsonArray jsonArr=resultJson.get("user_list").getAsJsonArray();
            String teacherAccount="null";
            Double score = 0.0;
            for(int i=0;i<jsonArr.size();i++){
                JsonObject jsob=jsonArr.get(i).getAsJsonObject();
                Double thisScore=jsob.get("score").getAsDouble();
                if(Constants.FACE_PASS_MIN_SCORE <= thisScore) {
                    if (thisScore >= score) {
                        teacherAccount = jsob.get("user_id").getAsString();
                        score = thisScore;
                    }
                }
            }
            System.out.println("LoginServiceImpl.faceLoginTeachBaidu,teacherAccount:" + teacherAccount + ",similar score:" + score);
            if (score >= Constants.FACE_PASS_MIN_SCORE) {
                Teacher teacher = teacherDao.selectByAccount(teacherAccount);
                if (teacher != null) {
                    return teacher;
                }
            }
        }
        return null;
    }

    @Override
    public int studentRegist(Student stu) {
        if(studentDao.insertSelective(stu)==1){
            return 1;
        }
        return 0;
    }

    private String generateNewUrl(String  baseData) throws Exception {
        System.out.println("LoginServiceImpl.sendPost");
        /**
         * 公共参数
         */
        //返回值的类型，支持 JSON 与 XML。默认为 XML,非必选。
        String  Format="json";
        //API 版本号，为日期形式：YYYY-MM-DD，本版本对应为 2018-12-03。
        String  Version="2018-12-03";
        //阿里云颁发给用户的访问服务所用的密钥 ID。
        String AccessKeyId= Constants.AK_ID;
        //签名方式，目前支持 HMAC-SHA1。
        String  SignatureMethod="HMAC-SHA1" ;
        /**
         * 请求的时间戳。日期格式按照 ISO8601 标准表示，并需要使用 UTC 时间。格式为：
          YYYY-MM-DDThh:mm:ssZ
         例如，2014-05-26T12:00:00Z（为北京时间 2014 年 5 月 26 日 12时 0 分 0 秒）
         */
        String  Timestamp= getDate.iSO8601Time();
        //签名算法版本，目前版本是 1.0。
        String  SignatureVersion="1.0";
        Integer SignatureNonce1=(int)(Math.random()*10000+10000);
        String SignatureNonce=SignatureNonce1.toString();
        System.out.println("LoginServiceImpl.sendPost,SignatureNonce:"+SignatureNonce);
        HashMap<String,String> params=new HashMap<>(12);
        params.put("Format",Format);
        params.put("Version",Version);
        params.put("AccessKeyId",AccessKeyId);
        params.put("SignatureMethod",SignatureMethod);
        params.put("Timestamp",Timestamp);
        params.put("SignatureVersion",SignatureVersion);
        params.put("SignatureNonce",SignatureNonce);
        params.put("Action","RecognizeFace");
        params.put("ImageUrl","http://a.com/a.jpg");
        //签名结果串，关于签名的计算方法，请参见 签名机制 。
        String Signature= SignatureUtils.generateSignature("POST",params,Constants.AK_SECRET);
        String newStr="https://face.cn-shanghai.aliyuncs.com/?Action=RecognizeFace";
        newStr+="&Format="+Format;
        newStr+="&Version="+Version;
        newStr+="&Signature="+Signature;
        newStr+="&AccessKeyId="+AccessKeyId;
        newStr+="&SignatureMethod="+SignatureMethod;
        newStr+="&Timestamp="+Timestamp;
        newStr+="&SignatureVersion="+SignatureVersion;
        newStr+="&SignatureNonce="+SignatureNonce;
        newStr+="&ImageUrl=http://a.com/a.jpg";
        System.out.println("LoginServiceImpl.generateNewUrl,newUrl:"+newStr);
        return newStr;
    }
    private String sendGet(String url){
        StringBuilder sb = new StringBuilder();
        try {
            URL realUrl = new URL(url);
            HttpURLConnection conn = (HttpURLConnection)realUrl.openConnection();
            //设置编码格式 解决中文乱码问题
            conn.setRequestMethod("POST");
            BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream(),"UTF-8"));
            String inputLine = null;
            while ( (inputLine = in.readLine()) != null) {
                sb.append(inputLine);
            }
            in.close();
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sb.toString();
    }

    /**
     * Format	String	否	返回值的类型，支持 JSON 与 XML。默认为 XML。
     Version	String	是	API 版本号，为日期形式：YYYY-MM-DD，本版本对应为 2018-12-03。
     AccessKeyId	String	是	阿里云颁发给用户的访问服务所用的密钥 ID。
     Signature	String	是	签名结果串，关于签名的计算方法，请参见 签名机制 。
     SignatureMethod	String	是	签名方式，目前支持 HMAC-SHA1。
     Timestamp	String	是	请求的时间戳。日期格式按照 ISO8601 标准表示，并需要使用 UTC 时间。格式为：
     YYYY-MM-DDThh:mm:ssZ
     例如，2014-05-26T12:00:00Z（为北京时间 2014 年 5 月 26 日 12时 0 分 0 秒）
     SignatureVersion	String	是	签名算法版本，目前版本是 1.0。
     SignatureNonce	String	是	唯一随机数，用于防止网络重放攻击。用户在不同请求间要使用不同的随机数值
     */


}
