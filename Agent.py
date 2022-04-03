import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from Buffer import ReplayBuffer
import os

def create_model(input_shape, fc1, fc2, actions, lr):
	model = tf.keras.Sequential()
	model.add(Dense(fc1, activation='relu', input_shape=input_shape))
	model.add(Dense(fc2, activation='relu'))
	model.add(Dense(actions, activation='linear'))

	model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
	return model

class Agent:
	def __init__(self, input_dim, n_actions, eps, eps_min, eps_decay, max_exp=1_00_000, fc1=11, fc2=256, 
				gamma=0.99, chckpt_dir="./Models", lr=0.005):

		self.input_dim = input_dim
		self.n_actions = n_actions

		self.eps = eps
		self.eps_decay = eps_decay
		self.eps_min = eps_min

		self.buffer = ReplayBuffer(max_exp, self.input_dim, self.n_actions)

		self.fc1 = fc1
		self.fc2 = fc2
		self.gamma = gamma
		self.chckpt_dir = chckpt_dir

		self.loss = tf.keras.losses.MeanSquaredError()

		if not os.path.isdir(self.chckpt_dir):
			os.mkdir(self.chckpt_dir)

		self.q_net = create_model(input_dim, self.fc1, self.fc2, n_actions, lr)
		self.target_net = create_model(input_dim, self.fc1, self.fc2, n_actions, lr)

	def remember(self, s, a, r, d, n_s):
		self.buffer.storeTransition(s, a, r, d, n_s)

	def train(self, batch_size):
		if self.buffer.count < batch_size:
			return

		s, a, r, d, n_s = self.buffer.getMiniBatch(batch_size)
		self.learn(s, a, r, d, n_s)

	def learn(self, s, a, r, d, n_s):

		if len(s.shape) == 1:
			s = tf.convert_to_tensor([s], dtype=tf.float32)
			n_s = tf.convert_to_tensor([n_s], dtype=tf.float32)
		
		next_q_vals = self.q_net.predict(n_s)

		target_q_vals = r + (self.gamma * np.max(next_q_vals, axis=-1) * (1-d))

		with tf.GradientTape() as tape:
			q_vals = self.q_net(s)
			one_hot_action = tf.keras.utils.to_categorical(a, num_classes=self.n_actions, dtype=np.float32)
			# print("Action a: ", a)
			# print("One hot _action: ", one_hot_action)

			Q = tf.reduce_sum(tf.multiply(q_vals, one_hot_action), axis=1)

			# print("Q-Val: ", Q, " || Target:- ", target_q_vals)

			loss = tf.keras.losses.MeanSquaredError()(Q, target_q_vals)

		gradient = tape.gradient(loss, self.q_net.trainable_variables)
		self.q_net.optimizer.apply_gradients(zip(gradient, self.q_net.trainable_variables))

	def reduceEps(self):
		self.eps = self.eps - self.eps_decay if self.eps > self.eps_min else self.eps_min

	def get_action(self, state):
		state = tf.convert_to_tensor([state], dtype=tf.float32)

		if np.random.rand() < self.eps:
			action = np.random.choice(self.n_actions)
		else:
			action = self.q_net.predict(state).argmax(axis=-1)[0]

		return action

	def save(self):
		self.q_net.save_weights(os.path.join(self.chckpt_dir, "./q_nets.h5"))
		self.target_net.save_weights(os.path.join(self.chckpt_dir, "./target_net.h5"))