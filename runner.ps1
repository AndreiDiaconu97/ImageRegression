#function train_base([string]$train_script, [string]$cfg_name, [string]$input_img, [string]$run_name, [int]$max_minutes)
#{
#    Write-Output "cfg_name : $cfg_name"
#    Write-Output "input_img : $input_img"
#    Write-Output "run_name : $run_name"
#    Write-Output "max_minutes : $max_minutes"
#
#    $cfg_flag = "--cfg-name"
#    $run_name_flag = "--run-name"
#    $img_flag = "--input-img"
#
#    if (!($cfg_name))
#    {
#        $cfg_flag = ""
#    }
#    if (!($input_img))
#    {
#        $img_flag = ""
#    }
#    if (!($run_name))
#    {
#        $run_name_flag = ""
#    }
#
#    python $train_script $cfg_flag $cfg_name $run_name_flag $run_name $img_flag $input_img --max-mins $max_minutes
#}
#
#Write-Output "Starting Experiments..."
#
##### Preliminary Studies BASE ####
#
#
##### TRAIN BASE ####
##train_base -train_script "src/reg_base.py"  -cfg_name "hparams_base_baseline" -input_img "data/mountains.jpg" -run_name "baseline_mountains" -max_minutes 0
##train_base -train_script "src/reg_base.py"  -cfg_name "hparams_base_baseline" -input_img "data/giraffe_low_poly.png" -run_name "baseline_lowpoly" -max_minutes 5
#
##### TRAIN ENSEMBLE ####
##train_base -train_script "src/reg_grownet.py"  -cfg_name "hparams_grownet_baseline" -input_img "data/mountains.jpg" -run_name "grownet_mountains" -max_minutes 5
##train_base -train_script "src/reg_grownet.py"  -cfg_name "hparams_grownet_baseline" -input_img "data/mountains.jpg" -run_name "grownet_mountains" -max_minutes 0
#train_base -train_script "src/reg_grownet.py"  -cfg_name "hparams_grownet_baseline" -input_img "data/mountains.jpg" -run_name "grownet_mountains_posOnlyFirst" -max_minutes 0

# IMAGES
$Mountains = "mountains_final.jpg"
$Giraffe = "giraffe_low_poly.png"
$Crowd = "crowd_generator.bmp"
$Unitn = "unitn.png"
$images = $Mountains, $Giraffe, $Crowd, $Unitn

# IMPLEMENTATIONS
$run_grownet = "python src/reg_grownet.py --max_mins 20"
$run_base = "python src/reg_base.py --max_mins 20"
#$run_xgboost = "python src/reg_xgboost.py --cfg_name hparams_xgboost_baseline --input_img data/$Mountains --max_mins 20"

# EXPERIMENTS

# Check if patience performance is affected by batch size
#Invoke-Expression $run_grownet" --run_name grownet_mountains_TESTplateau3"
#Invoke-Expression $run_base" --run_name referenceSirenOnly --hidden_layers 5 --hidden_size 256 --B_scale 0"
#Invoke-Expression $run_base" --run_name decay0.8"

Foreach ($image in $images)
{
    Invoke-Expression $run_base" --input_img data/$image --run_name reference_$image"
}




# TODO:
# try ensemble with NN