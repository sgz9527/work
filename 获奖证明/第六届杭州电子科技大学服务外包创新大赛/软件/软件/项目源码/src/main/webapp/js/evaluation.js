$(document).ready(function(){
    $("#updateEvaluationButton").click(function(){
        $("#weightArea").show();
        $("#evaluationdetail").attr("readonly",false);
        $("#evaluationdetail").css({
            outlineStyle:"solid",
            outlineColor:"orange",
            outlineWidth:"1px"
        });
    })
    $("#submitEvaluation").click(function(){
        var cW=parseFloat($('#evaluationChatWeight option:selected') .val());
        var tW=parseFloat($('#evaluationTestWeight option:selected') .val());
        var eW=parseFloat($('#evaluationExamWeight option:selected') .val());
        //alert("cw:"+cW+"|tw:"+tW+"|eW:"+eW);
        if((cW+tW+eW)==1.0){
            $("#courseIdButton").val(courseids);
            $.ajax({
                type: "GET",
                url: "updateEvaluation?t=" + new Date().getTime(),
                data:$("#updateEvalForm").serialize(),
                dataType: "json",
                cache: false,
                success: function (data) {
                    alert(data.result);
                    $("#updateEvaluationDiv").hide();
                    $("#updateEvaluationButton").attr("disabled",true);
                    $("#evaluationdetail").attr("readonly",true);
                },
                error: function (xhr) {
                    alert(xhr.responseText);
                }
            });
        }
        else alert("所有比重和必须为1！");

    })
})