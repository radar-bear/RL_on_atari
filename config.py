import argparse
import os

'''
Notes:

The whole reforcement learning system has several level.
The highest one is "period"
     we only test the model when a period ends by playing game "play_num" times
     the training will last "period_num" periods
then the "epoch"
     in each period we train the model "epoch_per_period" times
     the data pool won't change in a single epoch, but when an epoch ends, the data pool will be enqueued "sample_per_epoch" new observations.
          "data pool" is a FIFO_queue that contains many observations from env
          these observations are generate based on the present model
          and we sample batch from this data pool
the lowest is "step"
     in each epoch we train the model "step_per_epoch" times
     every step we usd SGD approach with "batch_size" to train the model

'''
cwd = os.getcwd()
parser = argparse.ArgumentParser()

# model parameters
parser.add_argument('--game', type=str, default='Breakout-v0',
                    help='Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='decay rate of past observations')
parser.add_argument('--epsilon', type=float, default=0.99,
                    help='the probability to try a random action but not the best action')
parser.add_argument('--enable_softmax_exploration', type=bool, default=False,
                    help='use softmax function to exploration or not')
parser.add_argument('--resize_width', type=int, default=80,
                    help='')
parser.add_argument('--resize_height', type=int, default=80,
                    help='')
parser.add_argument('--look_forward_step', type=int, default=4,
                    help='how many frames in a single input')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')

# periods parameter
parser.add_argument('--period_num', type=int, default=1000,
                    help='total period number')
parser.add_argument('--play_num', type=int, default=50,
                    help='play game several times to test the model')
parser.add_argument('--epoch_per_period', type=int, default=20,
                    help='epoch per period')
parser.add_argument('--sample_per_epoch', type=int, default=3000,
                    help='generate how many new records when a new epoch begins')
parser.add_argument('--step_per_epoch', type=int, default=500,
                    help='how many steps will an epoch last')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--log_step', type=int, default=50,
                    help='steps to print log of loss value')

# data pool parameters
parser.add_argument('--pool_max_len', type=int, default=5000,
                    help='the max record in a sample pool')

# dir parameters
parser.add_argument('--ckpt_dir', type=str, default=cwd+'/ckpt',
                    help='check point directory')
parser.add_argument('--record_dir', type=str, default=cwd+'/record',
                    help='game screen record directory')

args = parser.parse_args()

# game conf
keymap = {'Breakout-v0': [1, 4, 5]}

# additional parameters
args.actions = len(keymap[args.game])