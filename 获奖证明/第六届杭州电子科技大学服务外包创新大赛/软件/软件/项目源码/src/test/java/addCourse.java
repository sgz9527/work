import com.xqq.dao.CourseMapper;
import com.xqq.pojo.Course;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;

public class addCourse {
    @Resource
    CourseMapper courseDao;
    public static void main(String[] args) {
        addCourse ad=new addCourse();
      // ad.addCourses();
    }
    /*public void addCourses(){
        for(int i=0;i<50;i++)
        {
            Course course=new Course();
            int choose=(int)(1+Math.random()*6);
            switch(choose)
            {
                case 1:course.setCourseType("文科|外语");break;
                case 2:course.setCourseType("理科|地理|水土");break;
                case 3:course.setCourseType("工科|计算机|编程语言");break;
                case 4:course.setCourseType("文科|政治|思想理论");break;
                case 5:course.setCourseType("理科|数学|大学数学");break;
                case 6:course.setCourseType("工科|电子信息|电路基础");break;
            }
            course.setBelongSchId(2);
            course.setBelongSchName("某野鸡大学2");
            course.setChapterNum(i);
            course.setCourseIntroduce("aweoijfgaiwerhjgoiajwesidjfoiwajeoigjweofjawhejgaoieuwf");
            course.setCourseName("课程"+i);
            course.setTeacher("马云");
            course.setCoursePostSrc("13.jpg");
            course.setIntroduceVideoSrc("video1.mp4");
            course.setEvaluationLevel("ajwoiejgoiwejgiojaiwerfjuihwergoiajewidfgioaewjrgijewroig");
            course.setVideoNum(3);
            course.setCourseState("finish");
            course.setCourseProgress(i);
            courseDao.insert(course);
        }
    }*/
}
