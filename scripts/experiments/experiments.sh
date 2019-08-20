# nav0-baseline-td3-fcnet-1024
mapo --run OurTD3 --env Navigation-v0 --config-actor-net actor-1024-relu.json --config-critic-net critic-1024-relu.json --actor-lr 1e-4 --critic-lr 1e-4 --sample-batch-size 1 --train-batch-size 2048 --num-samples 5 --name nav0-baseline-td3-fcnet-1024

# nav0-mapo-true-dynamics-fcnet-1024
mapo --run MAPO --env Navigation-v0 --use-true-dynamics --config-actor-net actor-1024-relu.json --config-critic-net critic-1024-relu.json --actor-lr 1e-4 --critic-lr 1e-4 --sample-batch-size 1 --train-batch-size 2048 --num-samples 5 --name nav0-mapo-true-dynamics-fcnet-1024

# nav0-mapo-mle-fcnet-1024-linear-dynamics
mapo --run MAPO --env Navigation-v0 --model-loss mle --config-actor-net actor-1024-relu.json --config-critic-net critic-1024-relu.json --config-dynamics-net dynamics-linear-relu.json --actor-lr 1e-4 --critic-lr 1e-4 --dynamics-lr 1e-4 --sample-batch-size 1 --train-batch-size 2048 --num-samples 5 --name nav0-mapo-mle-fcnet-1024-linear-dynamics

# nav0-mapo-pga-fcnet-1024-linear-dynamics
mapo --run MAPO --env Navigation-v0 --model-loss pga --config-actor-net actor-1024-relu.json --config-critic-net critic-1024-relu.json --config-dynamics-net dynamics-linear-relu.json --actor-lr 1e-4 --critic-lr 1e-4 --dynamics-lr 1e-4 --sample-batch-size 1 --train-batch-size 2048 --num-samples 5 --name nav0-mapo-pga-fcnet-1024-linear-dynamics

# nav0-mapo-mle-fcnet-1024
mapo --run MAPO --env Navigation-v0 --model-loss mle --config-actor-net actor-1024-relu.json --config-critic-net critic-1024-relu.json --config-dynamics-net dynamics-1024-relu.json --actor-lr 1e-4 --critic-lr 1e-4 --dynamics-lr 1e-4 --sample-batch-size 1 --train-batch-size 2048 --num-samples 5 --name nav0-mapo-mle-fcnet-1024
