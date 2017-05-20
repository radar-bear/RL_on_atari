import cv2
import queue
import threading
from config import *
import numpy as np
import gym
from gym import wrappers
import tensorflow as tf

class data_pool():
    def __init__(self, max_len):
        self.max_len = max_len
        self._data = queue.Queue(max_len)
    def enqueue(self, element):
        self._data.put(element, timeout=10)
    def dequeue(self):
        return self._data.get(timeout=10)

def rgb2gray(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    return res

def resize(image):
    return cv2.resize(image, (args.resize_width, args.resize_height))

def generate_samples(pool, model, sess, final_epsilon):
    '''
    This function will interact with env and enqueue new batch
    into the pool

    Args:
        env -> an unwrapped gym env, because wrapped env will save a mp4
                file automatically
        pool -> a container for samples, type of data_pool
        sample_num -> new samples to generate

    Outputs:
        None, only operate the pool

    Note:
        env.next(action) return s at t+1 and reward at t
        every record should be (s_at_t,  r_after_action, action, s_at_t+1, end_flag)
    '''

    env = gym.make(args.game)
    # read the action map
    game_keymap = keymap[args.game]
    # total reward
    total_reward = 0.0
    game_num = 0
    # reset the game
    s = env.reset()
    s = rgb2gray(resize(s))
    s_series = np.stack([s for _ in range(args.look_forward_step)], axis=2)
    # reset the batch
    s_batch = []
    a_batch = []
    r_batch = []
    s_next_batch = []
    end_flag_batch = []
    # init epsilon
    epsilon = args.initial_epsilon
    unit_epsilon = (args.initial_epsilon-final_epsilon)/args.epsilon_anneal_frames
    # generate sample_num samples
    while True:
        # calculate probabilities for each actions
        # use the transposed s_series
        # p is the probability for exploration new actions
        if args.enable_softmax_exploration:
            p = model.get_softmax(sess, s_series)
        else:
            p = np.ones(args.actions)/args.actions
        # action selection
        # greater epsilon means more uncertainty and more exploration
        if np.random.sample()<epsilon:
            # choose an action from the keymap under probabilities p
            # this is Exploration
            action_index = np.random.choice(range(args.actions), p=p)
        else:
            # choose the best action
            # this is Exploitation
            action_index = np.argmax(p)
        action = game_keymap[action_index]
        # play and get returns
        s_next, reward, end_flag, _ = env.step(action)
        s_next = rgb2gray(resize(s_next))
        # make one-hot action
        one_hot_action = np.zeros(args.actions)
        one_hot_action[action_index] = 1.0
        # add reward to total_reward
        total_reward += reward
        # add to batch
        s_batch.append(s_series)
        a_batch.append(one_hot_action)
        r_batch.append(reward)
        s_series = np.append(s_next[:, :, np.newaxis], s_series[:, :, 0:3], axis=2)
        s_next_batch.append(s_series)
        end_flag_batch.append(end_flag)
        # if batch completed, enqueue
        if len(s_batch)>=args.batch_size:
            pool.enqueue([s_batch, r_batch, a_batch, s_next_batch, end_flag_batch])
            s_batch = []
            a_batch = []
            r_batch = []
            s_next_batch = []
            end_flag_batch = []
        # if game over, reset the game
        # and print the average score every several games
        if end_flag:
            game_num += 1
            if game_num%args.generator_log_step == 0:
                print("average_score=%.2f, epsilon=%.3f, final_epsilon=%.3f" %(total_reward/args.generator_log_step, epsilon, final_epsilon))
                total_reward = 0
            # reset s and s_series
            s = env.reset()
            s = rgb2gray(resize(s))
            # reset the s_series
            s_series = np.stack([s for _ in range(args.look_forward_step)], axis=2)
        # epsilon anneal
        if epsilon>final_epsilon:
            epsilon -= unit_epsilon

def test_model(env, model, sess):
    '''
    play the model args.play_num times and return the average score

    Args:
        env -> unwrapped gym env
        model -> QFuncModel
        sess -> tf.Session

    Outs:
        average score
    '''
    game_keymap = keymap[args.game]
    total_score = 0
    count = 0
    while count < args.play_num:

        # a new game begins
        # reset the game and get the first frame
        s = env.reset()
        s = rgb2gray(resize(s))
        # initialize the s_series
        s_series = []
        for _ in range(args.look_forward_step):
            s_series.append(s)

        while True:
            # get the action_index from model
            action_index = model.play(sess, np.transpose(np.array(s_series), [1,2,0]))
            action = game_keymap[action_index]

            # play one step
            s_next, reward, end_flag, _ = env.step(action)
            s_next = rgb2gray(resize(s_next))


            total_score += reward

            # is game over
            if end_flag:
                break

            # update the s_series
            s = s_next
            s_series.pop()
            s_series.insert(0, s)

        count += 1

    average_score = total_score/count

    return average_score

def record_play(env, model, sess, directory):
    '''
    The function will wrap the env, so the game screen will be recorded in a mp4 file. If you need to use the checkpoint, please restore the model before you call this function
    All files will be stored in dir
    '''

    game_keymap = keymap[args.game]
    # prepare the directory
    if tf.gfile.Exists(args.record_dir):
        tf.gfile.DeleteRecursively(args.record_dir)
    tf.gfile.MakeDirs(args.record_dir)
    # wrap the env
    env = wrappers.Monitor(env, directory, force=True)
    # game starts
    s = env.reset()
    s = rgb2gray(resize(s))
    # initialize the s_series
    s_series = []
    for _ in range(args.look_forward_step):
        s_series.append(s)

    isEnd = False
    while not isEnd:
        # get the action_index from model
        action_index = model.play(sess, np.transpose(np.array(s_series), [1,2,0]))
        action = game_keymap[action_index]

        # play one step
        s_next, reward, isEnd, _ = env.step(action)
        s_next = rgb2gray(resize(s_next))

        # update the s_series
        s = s_next
        s_series.pop()
        s_series.insert(0, s)

def restore_from(sess, saver, ckpt_dir, backNum=0):
    '''
    this function will read the ckpt_dir and restore the variables from the checkpoint automatically according to saver, which contains a dictionary for all variables to restore

    Args:
        sess -> the session to restore
        saver -> the variables to restore, generally the default saver
        ckpt_dir -> the parent directory of ckpt files
        backNum -> which history record you wanna roll back, 0 for the newest

    Outs:
        global_step -> the global step when the ckpt was made
    '''
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if len(ckpt.all_model_checkpoint_paths)<backNum+1:
        print("Back too many epochs")
        return -1
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.all_model_checkpoint_paths[backNum])
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    return global_step

def show_action_play(env, model, sess):
    '''
    Play the game once and print the actions
    '''

    game_keymap = keymap[args.game]

    # game starts
    s = env.reset()
    s = rgb2gray(resize(s))
    # initialize the s_series
    s_series = []
    for _ in range(args.look_forward_step):
        s_series.append(s)

    isEnd = False
    while not isEnd:
        # get the action_index from model
        action_index = model.play(sess, np.transpose(np.array(s_series), [1,2,0]))
        action = game_keymap[action_index]
        print(action)

        # play one step
        s_next, reward, isEnd, _ = env.step(action)
        s_next = rgb2gray(resize(s_next))

        # update the s_series
        s = s_next
        s_series.pop()
        s_series.insert(0, s)










