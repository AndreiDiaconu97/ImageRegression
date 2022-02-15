# IMAGES
$Mountains = "mountains_final.jpg"
$Giraffe = "giraffe_low_poly.png"
$Crowd = "crowd_generator.bmp"
$Unitn = "unitn.png"
$images = $Unitn # $Mountains, $Giraffe, $Crowd,

# IMPLEMENTATIONS
$run_base = "python src/reg_base.py --max_mins 20"
$run_grownet = "python src/reg_grownet.py --max_mins 20"
$run_xgboost = "python src/reg_xgboost.py --max_mins 20"

# EXPERIMENTS

# BASELINE IMPLEMENTATION #################################################################################################################
#Foreach ($image in $images)
#{
#    Invoke-Expression $run_base" --input_img data/$image --run_name reference_$image'_gradAcc' --acc_gradients True"
#    Invoke-Expression $run_base" --input_img data/$image --run_name reference_$image'_noPosEnc_100w0'   --w0 100 --B_scale 0"
#    Invoke-Expression $run_base" --input_img data/$image --run_name reference_$image'_noPosEnc_30w0'    --w0 30  --B_scale 0"
#    Invoke-Expression $run_base" --input_img data/$image --run_name reference_$image'_batch1000'        --batch_size 1000"
#    Invoke-Expression $run_base" --input_img data/$image --run_name reference_$image'_batch10000'       --batch_size 10000"
#    Invoke-Expression $run_base" --input_img data/$image --run_name reference_$image'_seqSampling'      --batch_sampling_mode sequence"
#}
###########################################################################################################################################

# GROWNET #################################################################################################################################
#Foreach ($image in $images)
#{
    #    Invoke-Expression $run_grownet" --input_img data/$image --run_name grownet_$image'_20corr_10weak_500patEns_50patWeak' --lr_patience_ensemble 500 --lr_patience_model 50"
    #    Invoke-Expression $run_grownet" --input_img data/$image --run_name grownet_$image'_10corr_20weak_1000patience_500cwd' --lr_patience_ensemble 1000 --lr_patience_model 1000 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 10 --epochs_per_stage 20"
    #    Invoke-Expression $run_grownet" --input_img data/$image --run_name grownet_$image'_20corr_10weak_500patience_500cwd' --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500  --epochs_per_correction 20 --epochs_per_stage 10"
    #    Invoke-Expression $run_grownet" --input_img data/$image --run_name grownet_$image'_5corr_10weak_1000patience_500cwd' --lr_patience_ensemble 1000 --lr_patience_model 1000 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 5 --epochs_per_stage 10"
    #    Invoke-Expression $run_grownet" --input_img data/$image --run_name grownet_$image'_5corr_10weak_500patience_500cwd' --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500  --epochs_per_correction 5 --epochs_per_stage 10"
    #    Invoke-Expression $run_grownet" --input_img data/$image --run_name grownet_$image'_10corr_20weak_1000patience_1000cwd' --lr_patience_ensemble 1000 --lr_patience_model 1000 --lr_cooldown_ensemble 1000 --lr_cooldown_weak 1000 --epochs_per_correction 10 --epochs_per_stage 20"
#    Invoke-Expression $run_grownet" --input_img data/$image --run_name grownet_$image'_20corr_200weak' --lr_patience_ensemble 100 --lr_patience_model 100 --lr_cooldown_ensemble 40 --lr_cooldown_weak 150 --epochs_per_correction 20 --epochs_per_stage 200"
#}
###########################################################################################################################################

# XGBOOST #################################################################################################################################
#Invoke-Expression $run_xgboost" --input_img data/$Mountains --run_name xgboost_$Mountains'_PSNR26' --desired_psnr 26"
#Invoke-Expression $run_xgboost" --input_img data/$Giraffe --run_name xgboost_$Giraffe'_PSNR40' --desired_psnr 40"
#Invoke-Expression $run_xgboost" --input_img data/$Crowd --run_name xgboost_$Crowd'_PSNR40' --desired_psnr 40"
#Invoke-Expression $run_xgboost" --input_img data/$Unitn --run_name xgboost_$Unitn'_PSNR81' --desired_psnr 81"
###########################################################################################################################################

# GROWNET ABLATION ########################################################################################################################
#### Corrective Steps #####
#Invoke-Expression $run_grownet" --input_img data/crowd_generator.bmp --run_name ablation_grownet_crowd_generator.bmp_20corr_20weak --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 20 --epochs_per_stage 20"
#Invoke-Expression $run_grownet" --input_img data/crowd_generator.bmp --run_name ablation_grownet_crowd_generator.bmp_10corr_20weak --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 10 --epochs_per_stage 20"
#Invoke-Expression $run_grownet" --input_img data/crowd_generator.bmp --run_name ablation_grownet_crowd_generator.bmp_1corr_20weak --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 1 --epochs_per_stage 20"
#Invoke-Expression $run_grownet" --input_img data/crowd_generator.bmp --run_name ablation_grownet_crowd_generator.bmp_0corr_20weak --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 0 --epochs_per_stage 20"

#### No Context Propagation ####
#Invoke-Expression $run_grownet" --input_img data/mountains_final.jpg --run_name ablation_grownet_mountains_final.jpg_noContext --lr_patience_ensemble 1000 --lr_patience_model 1000 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 10 --epochs_per_stage 20 --propagate_context False"
#Invoke-Expression $run_grownet" --input_img data/giraffe_low_poly.png --run_name ablation_grownet_giraffe_low_poly.png_noContext --lr_patience_ensemble 1000 --lr_patience_model 1000 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 5 --epochs_per_stage 10 --propagate_context False"
#Invoke-Expression $run_grownet" --input_img data/crowd_generator.bmp --run_name ablation_grownet_crowd_generator.bmp_noContext --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 20 --epochs_per_stage 10 --propagate_context False"
#Invoke-Expression $run_grownet" --input_img data/unitn.png --run_name ablation_grownet_unitn.png_noContext --lr_patience_ensemble 100 --lr_patience_model 100 --lr_cooldown_ensemble 40 --lr_cooldown_weak 150 --epochs_per_correction 20 --epochs_per_stage 200 --propagate_context False"

#### Use Boosting rate ####
#Invoke-Expression $run_grownet" --input_img data/mountains_final.jpg --run_name ablation_grownet_mountains_final.jpg_boostRate --lr_patience_ensemble 1000 --lr_patience_model 1000 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 10 --epochs_per_stage 20 --enable_boost_rate True"
#Invoke-Expression $run_grownet" --input_img data/giraffe_low_poly.png --run_name ablation_grownet_giraffe_low_poly.png_boostRate --lr_patience_ensemble 1000 --lr_patience_model 1000 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 5 --epochs_per_stage 10 --enable_boost_rate True"
#Invoke-Expression $run_grownet" --input_img data/crowd_generator.bmp --run_name ablation_grownet_crowd_generator.bmp_boostRate --lr_patience_ensemble 500 --lr_patience_model 500 --lr_cooldown_ensemble 500 --lr_cooldown_weak 500 --epochs_per_correction 20 --epochs_per_stage 10 --enable_boost_rate True"
#Invoke-Expression $run_grownet" --input_img data/unitn.png --run_name ablation_grownet_unitn.png_boostRate --lr_patience_ensemble 100 --lr_patience_model 100 --lr_cooldown_ensemble 40 --lr_cooldown_weak 150 --epochs_per_correction 20 --epochs_per_stage 200 --propagate_context True"
###########################################################################################################################################

#shutdown /s # WARNING!!!

## CREATE GIFS ##
#    magick convert 'out/base/reference_'$image'_seqSampling/error_*.png' 'out/base/reference_'$image'_seqSampling/_error.gif'
#    magick convert 'out/base/reference_'$image'_seqSampling/sample_*.png' 'out/base/reference_'$image'_seqSampling/_pred.gif'
