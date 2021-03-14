
    function uploadSucceed(msg){
        alert("succeed");
        $("#firstUploadSucceedMsg").html(msg);
        $("#firstUploadFailed").show();
        $("#firstUploadFailed").hide();
    }
    function uploadFailed(msg){
        alert("failed");
        $("#firstUploadFailedMsg").html(msg);
        $("#firstUploadFailed").hide();
        $("#firstUploadFailed").show();
    }
