function train_base([string]$train_script, [string]$cfg_name, [string]$input_img, [string]$run_name, [int]$max_minutes)
{
    Write-Output "cfg_name : $cfg_name"
    Write-Output "input_img : $input_img"
    Write-Output "run_name : $run_name"
    Write-Output "max_minutes : $max_minutes"

    $cfg_flag = "--cfg-name"
    $run_name_flag = "--run-name"
    $img_flag = "--input-img"

    if (!($cfg_name))
    {
        $cfg_flag = ""
    }
    if (!($input_img))
    {
        $img_flag = ""
    }
    if (!($run_name))
    {
        $run_name_flag = ""
    }

    python $train_script $cfg_flag $cfg_name $run_name_flag $run_name $img_flag $input_img --max-mins $max_minutes
}

Write-Output "Starting Experiments..."

#### Preliminary Studies BASE ####


#### TRAIN BASE ####
train_base -train_script "src/reg_base.py"  -cfg_name "hparams_base_baseline" -input_img "data/mountains.jpg" -run_name "baseline_mountains" -max_minutes 0
#train_base -train_script "src/reg_base.py"  -cfg_name "hparams_base_baseline" -input_img "data/giraffe_low_poly.png" -run_name "baseline_lowpoly" -max_minutes 5

#### TRAIN ENSEMBLE ####
#train_base -train_script "src/reg_grownet.py"  -cfg_name "hparams_grownet_baseline" -input_img "data/mountains.jpg" -run_name "grownet_mountains" -max_minutes 5



Write-Output "- Experiments Done -"
