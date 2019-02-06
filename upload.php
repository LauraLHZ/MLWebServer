<?php
$train_name=$_FILES['trainFile']['name'];
$train_tmp_name=$_FILES['trainFile']['tmp_name'];
$test_name=$_FILES['testFile']['name'];
$test_tmp_name=$_FILES['testFile']['tmp_name'];
if (is_uploaded_file($_FILES['trainFile']['tmp_name'])) {
    move_uploaded_file($train_tmp_name, 'upload/'.$train_name);
    //readfile($_FILES['trainFile']['tmp_name']);
}
if (is_uploaded_file($_FILES['testFile']['tmp_name'])) {
    move_uploaded_file($test_tmp_name, 'upload/'.$test_name);
}
/*if ($is_train_move && $is_test_move){
    echo "You have successfully uploading ".$train_name." and ".$test_name.".";
}*/
?>
