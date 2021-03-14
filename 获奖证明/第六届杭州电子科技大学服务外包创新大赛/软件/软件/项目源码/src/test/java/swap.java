import java.util.HashMap;

public class swap {
    public static void main(String[] args) {
        HashMap<String,Object> map1=new HashMap<String, Object>();
        map1.put("cnm",1);
        HashMap<String,Object> map2=new HashMap<String, Object>();
        map2.put("cnm",2);
        swap(map1,map2);
                        System.out.println("map1:"+map1.get("cnm"));
                        System.out.println("map2:"+map2.get("cnm"));
    }
    public static void swap(HashMap<String,Object> map1, HashMap<String,Object> map2){
                     HashMap<String,Object> temp=new HashMap<String, Object>();
        temp.putAll(map1);
        map1.putAll(map2);
        map2.putAll(temp);
    }
}
