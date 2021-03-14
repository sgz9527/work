package com.xqq.staticmethod;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

public class getDate {
    public static final long DAY=24L*60L*60L*1000L;
    public static Date getDate() {
        Date date = new Date();
        return date;
    }

    /**
     *
     * @param  before
     * @param  last
     * @return long
     */
    public static long DaysBetweenTwoDay(Date before,Date last)
    {
        long num=(last.getTime()-before.getTime())/DAY;
        System.out.println("getDate.DaysBetweenTwoDay,days:"+num);
        return num;
    }

    public static String toGMTString(Date date) {
        SimpleDateFormat sdf1 = new SimpleDateFormat("yyyy-MM-dd");
        SimpleDateFormat sdf2 = new SimpleDateFormat("HH:mm:ss");
        sdf2.setTimeZone(TimeZone.getTimeZone("UTC"));
        String result = sdf1.format(date) + "T" + sdf2.format(date) + "Z";
        return result;
    }
    public static String getYMD() {
        Date dateNow = new Date();
        SimpleDateFormat dt = new SimpleDateFormat("yyyy-MM-dd");
        //包含年月日的日期字符串
        String dateN = dt.format(dateNow);
        return dateN;
    }

    public static String getHMS() {
        Date dateNow = new Date();
        SimpleDateFormat dt = new SimpleDateFormat("HH:mm:ss");
        String dateN = dt.format(dateNow);//包含时分秒的日期字符串
        return dateN;
    }

    public static String getYMD(Date date) {
        SimpleDateFormat dt = new SimpleDateFormat("yyyy-MM-dd");
        String dateN = dt.format(date);//包含年月日的日期字符串
        return dateN;
    }

    public static String getHMS(Date date) {
        SimpleDateFormat dt = new SimpleDateFormat("HH:mm:ss");
        //包含时分秒的日期字符串
        String dateN = dt.format(date);
        return dateN;
    }
    public static String iSO8601Time(){
        System.out.println("getDate.iSO8601Time");
        String dateN=toGMTString(new Date());
        System.out.println("getDate.iSO8601Time:"+dateN);
        return dateN;

    }
}
