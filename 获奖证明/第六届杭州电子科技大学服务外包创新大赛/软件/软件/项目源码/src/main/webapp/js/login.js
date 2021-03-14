function Login() {
    if ($("#account").val() != ""&& $("#password").val() != "") {
        if ($('input[name="accountType"]:checked ').val() == "teacher") {
            $("#account").attr("name","teacherAccount");
            $("#password").attr("name","teacherPassword");
            $("#loginForm").attr("action", "teacherLogin");
            return true;
        }
        else if ($('input[name="accountType"]:checked ').val() == "student") {
            $("#account").attr("name","studentAccount");
            $("#password").attr("name","studentPassword");
            $("#loginForm").attr("action", "studentLogin");
            return true;
        }
    }
    alert("账号密码为必填项!");
    return false;
}