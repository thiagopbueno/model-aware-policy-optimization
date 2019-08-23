TEST_RUN=nav0/2019-08-23/run1
ENV=Navigation-v0
TIMESTEPS_TOTAL=100000
NUM_CPUS_FOR_DRIVER=1
NUM_GPUS=0.0
TRAIN_BATCH_SIZE=4096
NUM_SAMPLES=5
ACTOR_LR=1e-3
CRITIC_LR=1e-3
DYNAMICS_LR=1e-4
CONFIG_ACTOR_NET=fcnet-64-elu.json
CONFIG_CRITIC_NET=fcnet-64-elu.json


EXPERIMENT_LOG_DIR=~/ray_results/$TEST_RUN
EXPERIMENT_CONFIG_FILE=$EXPERIMENT_LOG_DIR/config.exp.txt
[[ -d $EXPERIMENT_LOG_DIR ]] || mkdir -p $EXPERIMENT_LOG_DIR
[[ -f $EXPERIMENT_CONFIG_FILE ]] || touch $EXPERIMENT_CONFIG_FILE
cat <<END >$EXPERIMENT_CONFIG_FILE
ENV = $ENV
TIMESTEPS_TOTAL = $TIMESTEPS_TOTAL
NUM_CPUS_FOR_DRIVER = $NUM_CPUS_FOR_DRIVER
NUM_GPUS = $NUM_GPUS
TRAIN_BATCH_SIZE = $TRAIN_BATCH_SIZE
NUM_SAMPLES = $NUM_SAMPLES
ACTOR_LR = $ACTOR_LR
CRITIC_LR = $CRITIC_LR
CONFIG_ACTOR_NET = $CONFIG_ACTOR_NET
CONFIG_CRITIC_NET = $CONFIG_CRITIC_NET
END


# EXPERIMENT=nav0-td3-baseline-fcnet-64
# mapo --run OurTD3 --env $ENV --config-actor-net $CONFIG_ACTOR_NET --config-critic-net $CONFIG_CRITIC_NET --actor-lr $ACTOR_LR --critic-lr $CRITIC_LR --train-batch-size $TRAIN_BATCH_SIZE --num-samples $NUM_SAMPLES --actor-delay 2 --timesteps-total $TIMESTEPS_TOTAL --name $TEST_RUN/$EXPERIMENT --num-cpus-for-driver $NUM_CPUS_FOR_DRIVER --num-gpus $NUM_GPUS --debug --evaluation-interval 5 --batch-mode truncate_episodes



# EXPERIMENT=nav0-mapo-true-dynamics-fcnet-64
# mapo --run MAPO --env $ENV --use-true-dynamics --config-actor-net $CONFIG_ACTOR_NET --config-critic-net $CONFIG_CRITIC_NET --actor-lr $ACTOR_LR --critic-lr $CRITIC_LR --train-batch-size $TRAIN_BATCH_SIZE --num-samples $NUM_SAMPLES --timesteps-total $TIMESTEPS_TOTAL --name $TEST_RUN/$EXPERIMENT --num-cpus-for-driver $NUM_CPUS_FOR_DRIVER --num-gpus $NUM_GPUS --debug


# EXPERIMENT=nav0-mapo-mle-fcnet-64-linear-dynamics
# mapo --run MAPO --env $ENV --model-loss mle --config-actor-net $CONFIG_ACTOR_NET --config-critic-net $CONFIG_CRITIC_NET --config-dynamics-net dynamics-linear-relu.json --actor-lr $ACTOR_LR --critic-lr $CRITIC_LR --dynamics-lr $DYNAMICS_LR --train-batch-size $TRAIN_BATCH_SIZE --num-samples $NUM_SAMPLES --timesteps-total $TIMESTEPS_TOTAL --name $TEST_RUN/$EXPERIMENT --num-cpus-for-driver $NUM_CPUS_FOR_DRIVER --num-gpus $NUM_GPUS --debug



EXPERIMENT=nav0-mapo-pga-fcnet-64-linear-dynamics
mapo --run MAPO --env $ENV --model-loss pga --config-actor-net $CONFIG_ACTOR_NET --config-critic-net $CONFIG_CRITIC_NET --config-dynamics-net dynamics-linear-relu.json --actor-lr $ACTOR_LR --critic-lr $CRITIC_LR --dynamics-lr $DYNAMICS_LR --train-batch-size $TRAIN_BATCH_SIZE --num-samples $NUM_SAMPLES --timesteps-total $TIMESTEPS_TOTAL --name $TEST_RUN/$EXPERIMENT --num-cpus-for-driver $NUM_CPUS_FOR_DRIVER --num-gpus $NUM_GPUS --debug

exit


EXPERIMENT=nav0-mapo-mle-fcnet-64
mapo --run MAPO --env $ENV --model-loss mle --config-actor-net $CONFIG_ACTOR_NET --config-critic-net $CONFIG_CRITIC_NET --config-dynamics-net fcnet-64-elu.json --actor-lr $ACTOR_LR --critic-lr $CRITIC_LR --dynamics-lr $DYNAMICS_LR --train-batch-size $TRAIN_BATCH_SIZE --num-samples $NUM_SAMPLES --timesteps-total $TIMESTEPS_TOTAL --name $TEST_RUN/$EXPERIMENT --num-cpus-for-driver $NUM_CPUS_FOR_DRIVER --num-gpus $NUM_GPUS --debug


EXPERIMENT=nav0-mapo-pga-fcnet-64
mapo --run MAPO --env $ENV --model-loss pga --config-actor-net $CONFIG_ACTOR_NET --config-critic-net $CONFIG_CRITIC_NET --config-dynamics-net fcnet-64-elu.json --actor-lr $ACTOR_LR --critic-lr $CRITIC_LR --dynamics-lr $DYNAMICS_LR --train-batch-size $TRAIN_BATCH_SIZE --num-samples $NUM_SAMPLES --timesteps-total $TIMESTEPS_TOTAL --name $TEST_RUN/$EXPERIMENT --num-cpus-for-driver $NUM_CPUS_FOR_DRIVER --num-gpus $NUM_GPUS --debug --num-sgd-iter 80


tensorboard --logdir=~/ray_results/$TEST_RUN
