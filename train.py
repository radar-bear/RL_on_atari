import tensorflow as tf
import numpy as np
import gym
import datetime
from reference import QFuncModel
from utils import *
from config import *

def train():
    # set initial environment
    env = gym.make(args.game)
    pool = data_pool('default_pool', args.pool_max_len)
    args.actions = len(keymap[args.game])

    with tf.Graph().as_default():

        model = QFuncModel(args)
        # model_training is the model to train
        # model is the model to calculate target Q-value
        model_training = QFuncModel(args)

        init_op = tf.group(tf.local_variables_initializer(),tf.global_variables_initializer())

        step = 0
        epoch = -1
        period = -1
        with tf.Session() as sess:
            # initialize all variables
            sess.run(init_op)

            # a new step starts
            while period < args.period_num:

                # wether a new epoch begin
                if step%args.step_per_epoch == 0:

                    # update the epoch num
                    epoch += 1
                    # update the sample pool every epoch
                    generate_samples(env, pool, args.sample_per_epoch, model, sess, args.epsilon)
                    print("%s epoch %d: reload sample pool, %d new records generated" % (datetime.datetime.now(), epoch, args.sample_per_epoch))
                    # update the model every epoch
                    model.copy(sess, model_training)

                    # wether a new period begin
                    if epoch%args.epoch_per_period == 0:

                        period += 1
                        average_score = test_model(env, model, sess)
                        print("%s period %d: average score %.2f" % (datetime.datetime.now(), period, average_score))

                # every step fetch a batch from pool
                s, r, a, s_next, isEnd = pool.next_batch(args.batch_size)

                # calculate the target Q-value using model
                # note: don't using model_training
                y = [r[i] if isEnd[i] else r[i]+args.gamma*model.maxQ(sess, s_next[i])
                        for i in range(args.batch_size)]

                # train the model_training
                sess.run(model_training.train_op,
                    feed_dict={ model_training.s:s,
                                model_training.a:a,
                                model_training.y:y } )

                step += 1

if __name__ == '__main__':
    train()
