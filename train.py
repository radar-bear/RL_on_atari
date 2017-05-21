import tensorflow as tf
import numpy as np
import gym
import datetime
import time
import threading
from reference import QFuncModel
from utils import *
from config import *

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def train():
    # the time that begin training
    train_begin = datetime.datetime.now()
    pool = data_pool(args.pool_max_len)
    with tf.Graph().as_default():

        model = QFuncModel(args)
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
        start_time = time.time()
        with tf.Session() as sess:
            # initialize all variables
            sess.run(init_op)

            # config threadings for enqueue data_pool
            async_lock = threading.Lock()
            epsilon_list = [0.5,0.5,0.1,0.1,0.01,0.01]
            threads = [threading.Thread(target=generate_samples,
                                        args=(pool, model, sess, epsilon))
                        for epsilon in epsilon_list]
            for t in threads:
                t.start()
            print("threads start...")
            # a new step starts
            while epoch < args.epoch_num:

                # save model when a new epoch begin
                if step%args.step_per_epoch == 0:
                    # update the epoch num
                    epoch += 1
                    saver.save(sess, args.ckpt_dir+'/Q-model', global_step=step)

                # every step fetch a batch from pool
                s, r, a, s_next, end_flag = pool.dequeue()
                # calculate the target Q-value using model
                # note: don't using model_training
                y = [r[i] if end_flag[i] else r[i]+args.gamma*model.maxQ(sess, s_next[i]) for i in range(len(s))]

                # train the model_trainingfdc
                # NOTE: in reinforcement learning, the loss value won't always decrease
                # but in general, it should be zero at last
                loss, _ = sess.run([model.loss, model.train_op],
                            feed_dict={ model.s:s,
                                        model.a:a,
                                        model.y:y } )
                # sess.run([model.train_op],
                #             feed_dict={ model.s:s,
                #                         model.a:a,
                #                         model.y:y } )

                if step%args.loss_log_step == 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    format_str = ('%s: step %d, loss = %.3e, %.3f frames/sec')
                    print(format_str % (datetime.datetime.now()-train_begin, step, loss, args.loss_log_step*args.batch_size/duration))

                step += 1

if __name__ == '__main__':
    train()
