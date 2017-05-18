import cv2
from config import *
import numpy as np
import gym
from gym import wrappers
import tensorflow as tf

class data_pool():
    def __init__(self, name, max_len, look_forward_step=4):
        self.name = name
        self.forward = look_forward_step
        self.max_len = max_len
        self._len = 0
        self._data = []
        # the record should be [s, r, a, isEnd]
        # where a is the index of action in neural network's output array
        # isEnd = True present this record is the end of a game
        self._need_to_shuffle = True
        self._start = 0
    def in_pool(self, element):
        self._data.append(element)
        self._need_to_shuffle = True
        if self._len >= self.max_len:
            del self._data[0]
            return
        self._len += 1
    def sample_one(self, index):
        '''
        Every record in a batch, the first element is a series of images
        it has shape [image_num, height, width]
        in tensorflow, the image_num generally handled just like channels
        so the user should transpose the tensor
        use np.transpose(x, [1, 2, 0]) to move the first index to the last

        the network need a series of images
        the first element is images at t+1
        and then images at t, t-1, t-2 ....
        '''
        if index+1 >= self._len:
            image_series = [self._data[index][0]]
        else:
            image_series = [self._data[index+1][0]]

        # to avoid given images between different game epoch
        # we check if the images belongs to a single game epoch
        # if not, break the filling process, and...
        for i in range(self.forward):
            if index-i<0:
                break
            elif self._data[index-i][-1]:
                break
            else:
                image_series.append(self._data[index-i][0])

        # ... fill the series by the last element of the series
        # the length of this series of images is 1 more than self.forward
        # because it contain the images at t+1
        for _ in range(self.forward+1-len(image_series)):
            image_series.append(image_series[-1])

        # return [ series_images, action, reward, the_next_s, is_end ]
        # the_next_s also a series of images
        # the flag "is_end" is used when calculating Q-value
        # if it is the end of a game, the Q-value just equals to reward
        return [image_series[1:], self._data[index][1], self._data[index][2], image_series[:-1], self._data[index][-1]]

    def next_batch(self, batch_size):
        # check is the batch_size is too big
        # if true, return all records and print a warning
        if batch_size > self.max_len:
            print("Batch size too big, return all records")
            return [self.sample_one(i) for i in range(self._len)]
        # check if the data_pool has been updated and thus need to be re-shuffled
        if self._need_to_shuffle or self._start + batch_size > self._len:
            self._perm =  np.arange(self._len)
            np.random.shuffle(self._perm)
            self._start = 0
        # batch is a list of sample_one()
        start = self._start
        end = start + batch_size
        self._start = end
        batch = [self.sample_one(i) for i in self._perm[start:end]]

        # process the batch
        s = [batch[i][0] for i in range(len(batch))]
        r = [batch[i][1] for i in range(len(batch))]
        a = [batch[i][2] for i in range(len(batch))]
        s_next = [batch[i][3] for i in range(len(batch))]
        isEnd = [batch[i][4] for i in range(len(batch))]

        # transpose the frames into shape [batch, height, width, frames]
        s = np.transpose(s, [0,2,3,1])
        s_next = np.transpose(s_next, [0,2,3,1])

        return [s, r, a, s_next, isEnd]

    def all_data(self):
        return [self.sample_one(i) for i in range(self._len)]

def rgb2gray(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    return res

def resize(image):
    return cv2.resize(image, (args.resize_width, args.resize_height))

def generate_samples(env, pool, sample_num, model, sess, epsilon):
    '''
    This function will interact with env and insert several new samples
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
        every record should be (s_at_t,  r_after_action, action, is_end_after_action)
    '''
    if sample_num > pool.max_len:
        print("Warning! Pool overflow, the oldest record will be deleted")
    # read the action map
    game_keymap = keymap[args.game]
    # count sample num
    count = 0
    # reset the game
    s = env.reset()
    s = rgb2gray(resize(s))
    s_series = []
    for _ in range(args.look_forward_step):
        s_series.append(s)
    # generate sample_num samples
    while count<sample_num:
        # calculate probabilities for each actions
        # use the transposed s_series
        # p is the probability for exploration new actions
        if args.enable_softmax_exploration:
            p = model.get_softmax(sess, np.transpose(np.array(s_series), [1,2,0]))
        else:
            p = np.ones(args.actions)/args.actions
        # action selection
        # greater epsilon means more uncertainty and more exploration
        if np.random.sample()<args.epsilon:
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
        # insert new record
        pool.in_pool([s, reward, one_hot_action, end_flag])
        # if game over, reset the game
        if end_flag:
            s = env.reset()
            s = rgb2gray(resize(s))
            # reset the s_series
            s_series = []
            for _ in range(args.look_forward_step):
                s_series.append(s)
        else:
            s = s_next
            # update the s_series
            s_series.pop()
            s_series.insert(0, s)

        count += 1

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











