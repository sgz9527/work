package com.xqq.staticmethod;

import com.xqq.constant.Constants;

import java.util.HashMap;
import java.util.Map;

/**
 * @author xqq
 */
public class faceRecongnize {

        /**
         * 重要提示代码中所需工具类
         * FileUtil,Base64Util,HttpUtil,GsonUtils请从
         * https://ai.baidu.com/file/658A35ABAB2D404FBF903F64D47C1F72
         * https://ai.baidu.com/file/C8D81F3301E24D2892968F09AE1AD6E2
         * https://ai.baidu.com/file/544D677F5D4E4F17B4122FBD60DB82B3
         * https://ai.baidu.com/file/470B3ACCA3FE43788B5A963BF0B625F3
         * 下载
         */
        public static String search(String baseData,String groupId) {
            // 请求url
            String url = "https://aip.baidubce.com/rest/2.0/face/v3/search";
            try {
                Map<String, Object> map = new HashMap<>();
                map.put("image", baseData);
                map.put("liveness_control", "NONE");
                map.put("group_id_list", groupId);
                //image_type include 3 types:BASE64,URL,FACE_TOKEN
                map.put("image_type", "BASE64");
                map.put("quality_control", "LOW");

                String param = GsonUtils.toJson(map);

                // 线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
                String accessToken = Constants.ACCESS_TOKEN;

                String result = HttpUtil.post(url, accessToken, "application/json", param);
                return result;
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }
        public static String add(String baseData,String groupId,String userAccount){
            // 请求url
            String url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add";
            try {
                Map<String, Object> map = new HashMap<>();
                map.put("image", baseData);
                map.put("group_id", groupId);
                map.put("user_id", userAccount);
                map.put("user_info", "very cool");
                map.put("liveness_control", "NONE");
                map.put("image_type", "BASE64");
                map.put("quality_control", "LOW");
                String param = GsonUtils.toJson(map);

                // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
                String accessToken=Constants.ACCESS_TOKEN;
                String result = HttpUtil.post(url, accessToken, "application/json", param);
                return result;
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }
}
