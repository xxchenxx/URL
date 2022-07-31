################### Training Single Domain Learning Networks ###################
function train_fn {
    CUDA_VISIBLE_DEVICES=<gpu id> python train_net.py --model.dir ./saved_results/sdl --model.name=$1 --data.train $2 --data.val $2 --data.test $2 --train.batch_size=$3 --train.learning_rate=$4 --train.max_iter=$5 --train.cosine_anneal_freq=$6 --train.eval_freq=$6
}

# Train an single domain learning network on every training dataset (the following models could be trained in parallel)

# ImageNet
NAME="imagenet-net"; TRAINSET="ilsvrc_2012"; BATCH_SIZE=64; LR="3e-2"; MAX_ITER=480000; ANNEAL_FREQ=48000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ