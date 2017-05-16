import tensorflow as tf
import numpy as np
import gym
from reference import QFuncModel
from utils import *
from config import *

def test():
    env = gym.make(args.game)
    with tf.Graph().as_default():
        model = QFuncModel(args)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=model.variable_list())
        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
        with tf.Session() as sess:
            for i in range(len(ckpt.all_model_checkpoint_paths)):
                restore_from(sess, saver, args.ckpt_dir, backNum=i)
                record_play(env, model, sess, args.record_dir)

if __name__ == '__main__':
    test()