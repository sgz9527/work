import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xqq.pojo.CheckCourse;

/**
 * @author xqq
 */
public class PassJson {
    public static void main(String[] args) {
        //String result="{\"error_code\":0,\"error_msg\":\"SUCCESS\",\"log_id\":304569285941765251,\"timestamp\":1558594176,\"cached\":0,\"result\":{\"face_token\":\"007cd6ca9fae681ff5ce7027aff4afee\",\"user_list\":[{\"group_id\":\"yunzhiAdminFace\",\"user_id\":\"123456xqq1\",\"user_info\":\"very cool\",\"score\":97.756317138672}]}}";
        CheckCourse cc=new CheckCourse();
        cc.setIntroduceSrc("aweg");
        cc.setTeacherId(1);
        cc.setTeacherPhone("122335");
        cc.setCheckId(1);
        cc.setCheckState("wait");
        cc.setCourseName("jisuanji");
        cc.setPosterSrc("wegr");
        cc.setBelongSchId(1);
        cc.setSuplement("wgwrhrehehtjjrthdttttttttttttttttttt");
        cc.setCourseIntroduce("waGHRwhhrtttyyyyyyyyyyyyyy");
        cc.setCourseType("com|2ase");
        PassJson pj=new PassJson();
        // ad.addCourses();
        JsonObject jsob=pj.claToJsonObject(cc);
        System.out.println("PassJson.main,jsob.get('courseName'):"+jsob.get("checkId"));
    }
    public void strPassToJson(String result){
        JsonObject jsonObj = null;
        System.out.println("create jsonObj:");
        JsonParser jspa=new JsonParser();
        JsonElement jsElement=jspa.parse(result);
        System.out.println("???");
        jsonObj = jsElement.getAsJsonObject();
//        System.out.println("jsonObjStr:"+jsonObj.getAsString() + "\n" );
        int errorCode=jsonObj.get("error_code").getAsInt();
        System.out.println("errorCode:"+errorCode);
    }
    public JsonObject claToJsonObject(CheckCourse cc){
        return new JsonParser().parse(new Gson().toJson(cc)).getAsJsonObject();
    }
}
