import com.xqq.service.CourseServiceImpl;

import java.util.ArrayList;

public class StringTest extends CourseServiceImpl {
    public static void main(String[] args) {
        StringTest st=new StringTest();
        ArrayList<String> strList=new ArrayList<String>();
        strList.add("aaa");
        strList.add("aba");
        strList.add("bbb");
        strList.add("ccc");
        strList.add("aaew");
        strList.add("cc");
        strList.add("ccc");
        strList.add("bab");
        strList.add("dab");
        strList.add("aaa");
        strList.add("bbb");
        strList.add("ccc");
        strList.add("bbb");
        strList.add("abc");
        strList.add("bbb");
        strList.add("aaa");
        strList.add("bbb");
        strList.add("bbb");
        strList.add("aab");
        strList.add("bbb");
        strList.add("acb");
        strList.add("bbb");
        strList.add("bbb");
        strList.add("bbb");
        strList.add("bbb");
        strList.add("aaaeege");
                   String str="数学|语文|英语|法语|计算机组成原理|数据结构";
        st.getBiggest(strList);
        st.segment(str,'|');
    }

}
