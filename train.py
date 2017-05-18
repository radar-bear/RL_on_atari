import tensorflow as tf
import numpy as np
import gym
import datetime
import time
from reference import QFuncModel
from utils import *
from config import *

def train():
    # set initial environment
    env = gym.make(args.game)
    pool = data_pool('default_pool', args.pool_max_len)
    final_epsilon = 0.5

    with tf.Graph().as_default():

        model = QFuncModel(args)
        # model_training is the model to train
        # model is the model to calculate target Q-value
        model_training = QFuncModel(args)
        # init op
        init_op = tf.group(tf.local_variables_initializer(),tf.global_variables_initializer())
        # prepare the directory and saver for checkpoint(ckpt)
        if tf.gfile.Exists(args.ckpt_dir):
            tf.gfile.DeleteRecursively(args.ckpt_dir)
        tf.gfile.MakeDirs(args.ckpt_dir)
        # only save the model, NOT the model_training
        saver = tf.train.Saver(var_list=model.variable_list())

        step = 0
        epoch = -1
        period = -1
        start_time = time.time()
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
                        # decrease the epsilon
                        args.epsilon -= 0.1*(args.epsilon - final_epsilon)
                        # test the model
                        average_score = test_model(env, model, sess)
                        print("%s period %d: average score %.2f" % (datetime.datetime.now(), period, average_score))
                        print("epsilon changed to %.2f" %args.epsilon)
                        saver.save(sess, args.ckpt_dir+'/Q-model', global_step=step)

                # every step fetch a batch from pool
                s, r, a, s_next, isEnd = pool.next_batch(args.batch_size)

                # calculate the target Q-value using model
                # note: don't using model_training
                y = [r[i] if isEnd[i] else r[i]+args.gamma*model.maxQ(sess, s_next[i])
                        for i in range(args.batch_size)]

                # train the model_training
                # NOTE: in reinforcement learning, the loss value won't always decrease
                # but in general, it should be zero at last
                loss, _ = sess.run([model_training.loss, model_training.train_op],
                            feed_dict={ model_training.s:s,
                                        model_training.a:a,
                                        model_training.y:y } )

                if step%args.log_step == 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    format_str = ('%s: step %d, loss = %.3e, %.3f samples/sec')
                    print(format_str % (datetime.datetime.now(), step, loss, args.log_step*args.batch_size/duration))

                step += 1

if __name__ == '__main__':
    train()
