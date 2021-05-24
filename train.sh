python Train.py \
  --name      test \
  --save_dir  ../Saved_Networks/tmp/ \
  --data_dir  ../Data/METData/ \
  --do_rot    True \
  --weight_to  0 \
  --weight_ratio 0.0 \
  --weight_shift 0.0 \
  --v_frac    0.5 \
  --n_ofiles  32 \
  --chnk_size 1024 \
  --bsize     1024 \
  --n_workers 4 \
  --depth     5 \
  --width     256 \
  --skips     0 \
  --nrm       True \
  --drpt      0.0 \
  --lr        1e-4 \
  --grad_clip 2 \
  --skn_weight 0 \
