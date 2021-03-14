package com.xqq.staticmethod;

/**
 * @author xqq
 */
import java.net.URLEncoder;
import java.util.Map;
public class UrlUtil {
    private final static String CHARSET_UTF8 = "utf8";
    /**
     *
     * @param url
     * @return
     */
    public static String urlEncode(String url) {
        if (!"".equals(url)) {
            try {
                url = URLEncoder.encode(url, "UTF-8");
            } catch (Exception e) {
                System.out.println("UrlUtil.urlEncode:error");
            }
        }
        return url;
    }
    public static String generateQueryString(Map<String, String> params, boolean isEncodeKV) {
        StringBuilder canonicalizedQueryString = new StringBuilder();
        for (Map.Entry<String, String> entry : params.entrySet()) {
            if (isEncodeKV) {
                canonicalizedQueryString.append(percentEncode(entry.getKey())).append("=")
                        .append(percentEncode(entry.getValue())).append("&");
            } else {
                canonicalizedQueryString.append(entry.getKey()).append("=")
                        .append(entry.getValue()).append("&");
            }
        }
        if (canonicalizedQueryString.length() > 1) {
            canonicalizedQueryString.setLength(canonicalizedQueryString.length() - 1);
        }
        return canonicalizedQueryString.toString();
    }
    public static String percentEncode(String value) {
        try {
            // 使用URLEncoder.encode编码后，将"+","*","%7E"做替换即满足 API规定的编码规范
            return value == null ? null : URLEncoder.encode(value, CHARSET_UTF8)
                    .replace("+", "%20").replace("*", "%2A").replace("%7E", "~");
        } catch (Exception e) {
            //不可能发生的异常
        }
        return "";
    }
}
