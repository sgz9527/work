package com.xqq.staticmethod;

import java.io.File;
import java.util.ArrayList;

public class Files {
    public static String isExistName(String path, String orginName)//返回重命名后的文件名
    {
        // System.out.println("Files.isExistName:orginName:"+orginName);
        //默认只有一级目录
        File file = new File(path);
        File[] fileArry = file.listFiles();
        ArrayList<String> fileNameList = new ArrayList<String>();
        for (int i = 0; i < fileArry.length; i++)//得到文件名集合,默认没有相同的
        {
            String fileName = fileArry[i].getName();
            // System.out.println("fileName:"+fileName);
            fileNameList.add(fileName);
        }
        String orginN = orginName;
        while (isExist(fileNameList, orginN)) {
            int num = (int) (Math.random() * 9);
            String orginPre = orginN.substring(0, orginN.lastIndexOf("."));
            String orginNext = orginN.substring(orginN.lastIndexOf("."), orginN.length());
            orginPre += num;//重新生成除去后缀的文件名
            // System.out.println("重新生成除去后缀的文件名:"+orginPre);
            orginN = orginPre + orginNext;
            //System.out.println("new orginName:"+orginN);
        }
        //System.out.println("final new orginName:"+orginN);
        return orginN;
    }

    public static boolean isExist(ArrayList<String> fileNameList, String fileName) {
        // System.out.println("Files.isExist:fileNameList.size():"+fileNameList.size()+"||fileName:"+fileName);
        for (int i = 0; i < fileNameList.size(); i++) {
            // System.out.println("fileNameList.get(i):"+fileNameList.get(i));
            if (fileName.equals(fileNameList.get(i))) {
                // System.out.println("exist one file with the same name ");
                return true;
            }
        }
        return false;
    }

    public static boolean deleteFile(String fileName) {
        File file = new File(fileName);
        // 如果文件路径所对应的文件存在，并且是一个文件，则直接删除
        if (file.exists() && file.isFile()) {
            //System.out.println("file is exist and format is correct");
            if (file.delete()) {
                System.out.println("删除单个文件" + fileName + "成功！");
                return true;
            } else {
                System.out.println("删除单个文件" + fileName + "失败！");
                return false;
            }
        } else {
            System.out.println("删除单个文件失败：" + fileName + "不存在！");
            return false;
        }
    }

    public static int deleteFile1(String fileName) {
        File file = new File(fileName);
        // 如果文件路径所对应的文件存在，并且是一个文件，则直接删除
        if (file.exists() && file.isFile()) {
            //System.out.println("file is exist and format is correct");
            if (file.delete()) {
                System.out.println("删除单个文件" + fileName + "成功！");
                return 1;
            } else {
                System.out.println("删除单个文件" + fileName + "失败！");
                return -1;
            }
        } else {
            System.out.println("删除单个文件失败：" + fileName + "不存在！");
            return 0;
        }
    }
}
