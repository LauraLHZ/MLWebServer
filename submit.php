<?php		
	$trainFile='upload/'.$_POST['trainFile'];
	echo($trainFile);
	$testFile='upload/'.$_POST['testFile'];
	echo($trainFile);
	$mltype=$_POST['mltype'];
	echo($mltype);
	shell_exec('cd ~/anaconda3/bin');
	//shell_exec("python3 main.py upload/skin_trn.csv upload/skin_tst.csv knn");
	shell_exec("python3 main.py $trainFile $testFile $mltype");
?>
	