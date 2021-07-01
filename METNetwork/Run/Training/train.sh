python Train.py \
  --name      normal \
  --save_dir  /home/matthew/Documents/PhD/Saved_Networks/tmp/ \
  --data_dir  /home/matthew/Documents/PhD/Data/METData/Rotated/ \
  --v_frac    0.5 \
  --n_ofiles  32 \
  --chnk_size 1024 \
  --weight_to  300 \
  --weight_ratio 0.0 \
  --weight_shift 0.0 \
  --act       lrlu \
  --depth     5 \
  --width     256 \
  --nrm       True \
  --drpt      0.0 \
  --loss_nm   hbloss \
  --opt_nm    adam \
  --lr        1e-4 \
  --grad_clip 2 \
  --skn_weight 0.5 \
  --b_size    1024 \
  --n_workers 0 \