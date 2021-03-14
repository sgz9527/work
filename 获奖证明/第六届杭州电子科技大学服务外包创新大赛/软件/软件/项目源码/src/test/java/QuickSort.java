import com.xqq.service.CourseServiceImpl;

import java.util.ArrayList;
import java.util.HashMap;

public class QuickSort extends CourseServiceImpl {
    public static void main(String[] args) {
        ArrayList<HashMap<String,Object>> mapList=new ArrayList<HashMap<String, Object>>();
        HashMap<String,Object> map1=new HashMap<String, Object>();
        map1.put("fans",1);
        HashMap<String,Object> map2=new HashMap<String, Object>();
        map2.put("fans",10);
        HashMap<String,Object> map3=new HashMap<String, Object>();
        map3.put("fans",13);
        HashMap<String,Object> map4=new HashMap<String, Object>();
        map4.put("fans",15);
        HashMap<String,Object> map5=new HashMap<String, Object>();
        map5.put("fans",5);
        HashMap<String,Object> map6=new HashMap<String, Object>();
        map6.put("fans",3);
        HashMap<String,Object> map7=new HashMap<String, Object>();
        map7.put("fans",46);
        HashMap<String,Object> map8=new HashMap<String, Object>();
        map8.put("fans",21);
        HashMap<String,Object> map9=new HashMap<String, Object>();
        map9.put("fans",3);
        HashMap<String,Object> map10=new HashMap<String, Object>();
        map10.put("fans",2);
        mapList.add(map1);
        mapList.add(map2);
        mapList.add(map3);
        mapList.add(map4);
        mapList.add(map5);
        mapList.add(map6);
        mapList.add(map7);
        mapList.add(map8);
        mapList.add(map9);
        mapList.add(map10);
        for(int i=0;i<mapList.size();i++)
            System.out.println("before:"+mapList.get(i).get("fans"));
        QuickSort qc=new  QuickSort();
        qc.sort(mapList,0,mapList.size()-1);
               for(int i=0;i<mapList.size();i++)
                   System.out.println(mapList.get(i).get("fans"));
    }

}
