import numpy as np

class ReplayBuffer():
	def __init__(self, max_exp, input_dims, n_actions=3):

		self.max_exp = max_exp

		self.state = np.empty((self.max_exp, *input_dims), dtype=np.float32)
		self.action = np.empty(self.max_exp, dtype=np.float32)
		self.reward = np.empty(self.max_exp, dtype=np.float32)
		self.done = np.empty(self.max_exp, dtype='int')
		self.next_state = np.empty((self.max_exp, *input_dims), dtype=np.float32)

		self.count = 0

	def storeTransition(self, s, a, r, d, n_s):
		index = self.count % self.max_exp

		self.state[index] = s
		self.action[index] = a
		self.reward[index] = r
		self.done[index] = d
		self.next_state[index] = n_s

		self.count += 1

	def getMiniBatch(self, batch_size):

		if self.count < batch_size:			
			return

		batch_mem = min(self.count, self.max_exp)
		batch = np.random.choice(batch_mem, size=batch_size)

		s = self.state[batch]
		a = self.action[batch]
		r = self.reward[batch]
		d = self.done[batch].astype('float')
		s_ = self.next_state[batch]

		return s, a, r, d, s_