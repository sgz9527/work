<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head >
    <script src="js/jquery-2.0.3.js"></script>
    <title>loginPage</title>
    <script>
        $(document).ready(function(){
            $("#button1").click(function(){
                $.ajax({
                    type:"POST",
                    url:"teacherLogin",
                    data:$("#form1").serialize(),
                    cache:false,
                    dataType:"json",
                    success:function(msg){
                        alert(msg.LoginTeacherInfo);
                    },
                    error:function(e){
                        alert(e.responseText);
                    }
                })
            })
        });
    </script>
</head>
<body>
<form action=""  id="form1">
    account:<input type="text" name="TeacherAccount" ><br>
    password:<input type="password" name="TeacherPassword">
    <br>
    <input type="button" value="点击登录" id="button1" >
</form>
</body>
</html>

