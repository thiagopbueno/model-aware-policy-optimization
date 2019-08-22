# nav0-td3-baseline-fcnet-64
mapo --run OurTD3 --env Navigation-v0 --config-actor-net fcnet-64-elu.json --config-critic-net fcnet-64-elu.json --actor-lr 1e-4 --critic-lr 1e-4 --train-batch-size 128 --num-samples 5 --actor-delay 2 --timesteps-total 200000 --name nav0-td3-baseline-fcnet-64 --num-cpus-for-driver 2 --num-gpus 1.0 --debug

# nav0-mapo-true-dynamics-fcnet-64
mapo --run MAPO --env Navigation-v0 --use-true-dynamics --config-actor-net fcnet-64-elu.json --config-critic-net fcnet-64-elu.json --actor-lr 1e-4 --critic-lr 1e-4 --train-batch-size 128 --num-samples 5 --timesteps-total 200000 --name nav0-mapo-true-dynamics-fcnet-64 --num-cpus-for-driver 2 --num-gpus 1.0 --debug

# nav0-mapo-mle-fcnet-64-linear-dynamics
mapo --run MAPO --env Navigation-v0 --model-loss mle --config-actor-net fcnet-64-elu.json --config-critic-net fcnet-64-elu.json --config-dynamics-net dynamics-linear-relu.json --actor-lr 1e-4 --critic-lr 1e-4 --dynamics-lr 1e-4 --train-batch-size 128 --num-samples 5 --timesteps-total 200000 --name nav0-mapo-mle-fcnet-64-linear-dynamics --num-cpus-for-driver 2 --num-gpus 1.0 --debug

# nav0-mapo-pga-fcnet-64-linear-dynamics
mapo --run MAPO --env Navigation-v0 --model-loss pga --config-actor-net fcnet-64-elu.json --config-critic-net fcnet-64-elu.json --config-dynamics-net dynamics-linear-relu.json --actor-lr 1e-4 --critic-lr 1e-4 --dynamics-lr 1e-4 --train-batch-size 128 --num-samples 5 --timesteps-total 200000 --name nav0-mapo-pga-fcnet-64-linear-dynamics --num-cpus-for-driver 2 --num-gpus 1.0 --debug

# nav0-mapo-mle-fcnet-64
mapo --run MAPO --env Navigation-v0 --model-loss mle --config-actor-net fcnet-64-elu.json --config-critic-net fcnet-64-elu.json --config-dynamics-net fcnet-64-elu.json --actor-lr 1e-4 --critic-lr 1e-4 --dynamics-lr 1e-4 --train-batch-size 128 --num-samples 5 --timesteps-total 200000 --name nav0-mapo-mle-fcnet-64 --num-cpus-for-driver 2 --num-gpus 1.0 --debug

# nav0-mapo-pga-fcnet-64
mapo --run MAPO --env Navigation-v0 --model-loss pga --config-actor-net fcnet-64-elu.json --config-critic-net fcnet-64-elu.json --config-dynamics-net fcnet-64-elu.json --actor-lr 1e-4 --critic-lr 1e-4 --dynamics-lr 1e-4 --train-batch-size 128 --num-samples 5 --timesteps-total 200000 --name nav0-mapo-pga-fcnet-1024 --num-cpus-for-driver 2 --num-gpus 1.0 --debug
