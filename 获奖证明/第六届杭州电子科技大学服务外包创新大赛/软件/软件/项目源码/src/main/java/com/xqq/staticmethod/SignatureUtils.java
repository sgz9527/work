package com.xqq.staticmethod;

/**
 * @author xqq
 */

import sun.misc.BASE64Encoder;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.Map;
import java.util.TreeMap;
public class SignatureUtils {
    private final static String CHARSET_UTF8 = "utf8";
    private final static String ALGORITHM = "UTF-8";
    private final static String SEPARATOR = "&";

    /**
     * url:带有参数的原网址
     * @param url
     * @return
     * @throws URISyntaxException
     * @throws UnsupportedEncodingException
     */
    public static Map<String, String> splitQueryString(String url)
            throws URISyntaxException, UnsupportedEncodingException {
        URI uri = new URI(url);
        String query = uri.getQuery();
        final String[] pairs = query.split("&");
        TreeMap<String, String> queryMap = new TreeMap<String, String>();
        for (String pair : pairs) {
            final int idx = pair.indexOf("=");
            final String key = idx > 0 ? pair.substring(0, idx) : pair;
            if (!queryMap.containsKey(key)) {
                queryMap.put(key, URLDecoder.decode(pair.substring(idx + 1), CHARSET_UTF8));
            }
        }
        return queryMap;
    }
    /**
     * method：get or post
     * parameter:网址后面的参数数组
     * accessKeySecret:用户秘钥
     * @param method
     * @param parameter
     * @param accessKeySecret
     * @return 返回最终的签名字符串
     */
    public static String generateSignature(String method, Map<String, String> parameter,
                                  String accessKeySecret) throws Exception {
        System.out.println("SignatureUtils.generateSignature");
        String signString = generateSignString(method, parameter);
        System.out.println("signString---"+signString);
        byte[] signBytes = hmacSHA1Signature(accessKeySecret + "&", signString);
        String signature = newStringByBase64(signBytes);
        String signatureNew=URLEncoder.encode(signature, "UTF-8");
        System.out.println("signature---"+signatureNew);
        return signatureNew;
    }
    public static String generateSignString(String httpMethod, Map<String, String> parameter)
            throws IOException {
        System.out.println("SignatureUtils.generateSignString");
        TreeMap<String, String> sortParameter = new TreeMap<>();
        sortParameter.putAll(parameter);
        String canonicalizedQueryString = UrlUtil.generateQueryString(sortParameter, true);
        if (null == httpMethod) {
            throw new RuntimeException("httpMethod can not be empty");
        }
        StringBuilder stringToSign = new StringBuilder();
        stringToSign.append(httpMethod).append(SEPARATOR);
        stringToSign.append(percentEncode("/")).append(SEPARATOR);
        stringToSign.append(percentEncode(canonicalizedQueryString));
        return stringToSign.toString();
    }
    public static String percentEncode(String value) {
        try {
            return value == null ? null : URLEncoder.encode(value, CHARSET_UTF8)
                    .replace("+", "%20").replace("*", "%2A").replace("%7E", "~");
        } catch (Exception e) {
        }
        return "";
    }
    public static byte[] hmacSHA1Signature(String secret, String baseString)
            throws Exception {
        if ("".equals(secret)) {
            throw new IOException("secret can not be empty");
        }
        if ("".equals(baseString)) {
            return null;
        }
        Mac mac = Mac.getInstance("HmacSHA1");
        SecretKeySpec keySpec = new SecretKeySpec(secret.getBytes(CHARSET_UTF8), ALGORITHM);
        mac.init(keySpec);
        return mac.doFinal(baseString.getBytes(CHARSET_UTF8));
    }

    /**
     * 将得到的签名字符串进行base64编码
     * @param bytes
     * @return
     * @throws UnsupportedEncodingException
     */
    public static String newStringByBase64(byte[] bytes)
            throws UnsupportedEncodingException {
        if (bytes == null || bytes.length == 0) {
            return null;
        }
        BASE64Encoder encoder=new BASE64Encoder();
        return encoder.encode(bytes);
    }


}
