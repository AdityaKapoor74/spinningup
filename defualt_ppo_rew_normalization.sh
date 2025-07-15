# Run each seed separately
for seed in 0 10 20 30 40; do
    echo "Running seed $seed..."
    python -m spinup.run ppo \
        --env Walker2d-v2 \
        --normalize_rew \
        --clip_rew 10.0 \
        --exp_name walker2d_rew_norm_seed_$seed \
        --epochs 300 \
        --seed $seed \
        --data_dir ./data \
        --steps_per_epoch 4000
done