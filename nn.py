import tensorflow as tf
import numpy as np
from typing import Tuple
from tqdm import tqdm, trange
from training_data import TrainingData
from termcolor import colored

class NN:
    @staticmethod
    def CNN_layer(
            input_ph,
            window_shape,
            output_dim: int,
            strides,
            use_maxpool: bool = False,
        ):
        '''
        input_ph: the input variable
        window_shape: H × W of the window
        output_dim: the dim of output
        strides: the skip size of sliding window
        use_maxpool: to use max pooling or not
        '''
        input_ch = input_ph.get_shape()[-1].value
        W = tf.Variable(tf.truncated_normal(shape=[*window_shape, input_ch, output_dim], stddev=0.1),
                        name='W')
        b = tf.Variable(tf.zeros(shape=[output_dim]), name='b')
        if use_maxpool:
            conv = tf.nn.relu(
                tf.nn.conv2d(input_ph, W, strides=(1,1,1,1), padding='SAME') + b)
            output = tf.nn.max_pool(conv, ksize=strides, strides=strides, padding='SAME')
        else:
            output = tf.nn.relu(
                tf.nn.conv2d(input_ph, W, strides=strides, padding='SAME') + b)
        return output

    @staticmethod
    def ReLu_layer(
            input_ph,
            output_dim: int,
        ):
        # The dim of input is batch_size × input_w
        input_w = input_ph.get_shape()[1].value

        W = tf.Variable(
            tf.truncated_normal((input_w, output_dim),
                                stddev=1.0/np.sqrt(input_w+output_dim)),
            name='W'
        )

        b = tf.Variable(tf.zeros(shape=[output_dim]), name='b')
        output = tf.nn.relu(tf.matmul(input_ph, W) + b)
        return output

    def __init__(self,
                 input_shape: np.ndarray,
                 action_n: int,
                 batch_size: int,) -> None:

        '''
        input_shape: the shape of input
        action_n: the number of action
        batch_size: batch_size
        '''

        self.batch_size = batch_size
        self.input_shape = input_shape

        print(colored('[Building graph]', 'cyan'), 'start building...')

        # The place holder of input (feed from here...)
        # Dim = batch_size × V1 × V2 ...
        self.input_ph = tf.placeholder(
            tf.float32, shape=(batch_size, *input_shape))

        hidden_layers = []

        with tf.variable_scope('conv0'):
            hidden_layers.append(self.CNN_layer(
                self.input_ph, # input var
                (4, 4),
                10, # output dim
                (1, 3, 3, 1), # strides
            ))

        with tf.variable_scope('conv1'):
            hidden_layers.append(self.CNN_layer(
                hidden_layers[-1], # input var
                (3, 3),
                5, # output dim
                (1, 2, 2, 1), # strides
            ))

        # Flatten the output
        hidden_layers.append(tf.reshape(hidden_layers[-1], (batch_size, -1)))

        with tf.variable_scope('hid0'):
            hidden_layers.append(self.ReLu_layer(
                hidden_layers[-1], # input var
                100, # output dim
            ))

        with tf.variable_scope('hid1'):
            hidden_layers.append(self.ReLu_layer(
                hidden_layers[-1], # input var
                30, # output dim
            ))

        with tf.variable_scope('output'):
            hidden_layers.append(self.ReLu_layer(
                hidden_layers[-1], # input var
                action_n, # output dim
            ))

        self.output = hidden_layers[-1]
        self.output_action = tf.argmax(self.output, axis=1)

        self.target_action_ph = tf.placeholder(tf.int32, shape=(batch_size,))
        self.target_value_ph = tf.placeholder(tf.float32, shape=(batch_size,))
        target_action_one_hot = tf.one_hot(self.target_action_ph, depth=action_n, dtype=tf.float32)
        output_mult_mask = tf.reduce_sum(self.output * target_action_one_hot, 1)
        self.loss = tf.nn.l2_loss(output_mult_mask - self.target_value_ph, name="l2_loss")
        self.eta_ph = tf.placeholder(tf.float32, shape=[])

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.eta_ph,
            momentum=0.1,
        )

        self.train_op = optimizer.minimize(self.loss)

        print(colored('[Building graph]', 'cyan'), 'Graph build:')
        print(colored('[Summary]', 'green'), ' → '.join(
            '(' + '×'.join(str(x) for x in ly.get_shape()) + ')' for ly in hidden_layers
        ))


        self.initialize()

    def initialize(self) -> None:
        # Initialize variables
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session.run(init)

    def train(self,
              training_data: TrainingData,
              epoch: int,
              eta: float):

        print(colored('[NNet Training]', 'cyan'), f'Start NN training..., η = {eta:.4f}')

        data_n = training_data.n

        with tqdm(total=epoch*len(training_data)) as tbar:
            for i in range(epoch):
                loss_tot = 0.
                for batch in training_data:
                    feed_dict = {
                        self.input_ph: batch[0],
                        self.target_action_ph: batch[1],
                        self.target_value_ph: batch[2],
                        self.eta_ph: eta,
                    }
                    _, loss_value = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
                    loss_tot += loss_value
                    tbar.update(1)

                if i % 4 == 0 or i == epoch-1:
                    tbar.write(colored(f"Epoch {i:2}: ", 'green') + f'loss = {loss_tot/training_data.real_n:.4f}')
                    tbar.update(0)

    def _feed(self, input_dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Private function for feeding an input.
        input_dt: the input (observation)
        return: (actions, Q values)
        '''
        input_n = input_dt.shape[0]
        assert input_n <= self.batch_size

        # If the input is less than batch_size, fill the remaining with 0
        input_dt = np.concatenate([
            input_dt,
            np.zeros((self.batch_size-input_n, *self.input_shape)),
        ])

        feed_dict = {
            self.input_ph: input_dt
        }
        res = self.session.run([self.output_action, self.output], feed_dict=feed_dict)
        return (res[0][:input_n], res[1][:input_n])

    def feed(self, input_dt) -> Tuple[np.ndarray, np.ndarray]:
        '''Feeding an input.
        input_dt: the input (observation), could be a single input or a np.array of input
        return: (actions, Q values)
        '''

        # Force input_dt to be numpy.array
        input_dt = np.array(input_dt)

        # If input_dt is a single input, wrap it with np.array
        if input_dt.ndim == len(self.input_shape):
            a, v = self.feed(np.array([input_dt]))
            return a[0], v[0]

        n = input_dt.shape[0]

        res = tuple(zip(
            *(self._feed(input_dt[s : s+self.batch_size])
              for s in range(0, n, self.batch_size))
        ))
        return (np.concatenate(res[0]), np.concatenate(res[1]))
