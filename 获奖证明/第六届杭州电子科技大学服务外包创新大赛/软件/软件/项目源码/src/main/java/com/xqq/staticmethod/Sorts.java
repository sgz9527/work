package com.xqq.staticmethod;

import java.util.ArrayList;
import java.util.HashMap;

public class Sorts {
    public static void quickSort(ArrayList<HashMap<String, Object>> mapList, String key, int low, int high) {
        int pivot;
        if (low < high) {
            pivot = findPivot(mapList, key, low, high);
            quickSort(mapList, key, low, pivot - 1);//左边序列。第一个索引位置到关键值索引-1
            quickSort(mapList, key, pivot + 1, high);//右边序列。从关键值索引+1到最后一个
        }
    }

    public static int findPivot(ArrayList<HashMap<String, Object>> mapList2, String key1, int low, int high) {
        HashMap<String, Object> map1 = new HashMap<String, Object>();
        map1.putAll(mapList2.get(low));
        int key = Integer.parseInt(mapList2.get(low).get(key1).toString());//得到排序的关键字
        //System.out.println("courseServiceImpl.QuickSort:fans:"+key);
        while (low < high) {
            while (low < high && Integer.parseInt(mapList2.get(high).get(key1).toString()) <= key)  //如果没有比关键值小的，比较下一个，直到有比关键值小的交换位置，然后又从前往后比较
                high--;
            mapList2.get(low).putAll(mapList2.get(high));
            while (low < high && Integer.parseInt(mapList2.get(low).get(key1).toString()) >= key)//如果没有比关键值大的，比较下一个，直到有比关键值大的交换位置
                low++;
            mapList2.get(high).putAll(mapList2.get(low));
        }
        // System.out.println("?? "+low);
        // System.out.println("find pivot:"+map1.get("fans"));
        mapList2.get(low).putAll(map1);
        // System.out.println("map1:"+map1.get("fans"));
        //  System.out.println("mapList2.get(low).get(fans):"+mapList2.get(low).get("fans")+" low:"+low);
        return low;
        //递归
    }
}
