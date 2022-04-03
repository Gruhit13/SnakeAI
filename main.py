from SnakeGameEnv import SnakeGameEnv
from Agent import Agent
import numpy as np
import time

def main():
	env = SnakeGameEnv()
	
	OBS_SHAPE = env.getObservationShape()
	N_ACTION = 3

	BATCH_SIZE = 1000
	EPS = 1.0
	EPS_MIN = 0.05
	EPS_DECAY = 0.01
	EPS_STEP = 1

	LEARNING_RATE = 0.001
	agent = Agent(OBS_SHAPE, N_ACTION, EPS, EPS_MIN, EPS_DECAY, lr=LEARNING_RATE)

	EPISODES = 300
	SAVE_AFTER = 50

	score_list = []
	best_score = 0	 

	for e in range(EPISODES):

		obs = env.reset()
		done = False
		
		episodic_reward = 0

		while not done:
			action = agent.get_action(obs)
			
			# print("Action:- ", action)
			next_obs, reward, done, _ = env.step(action)
			episodic_reward += reward
			
			agent.remember(obs, action, reward, done, next_obs)
			agent.learn(obs, action, reward, done, next_obs)	#	Train model per step

			obs = np.copy(next_obs)

		agent.train(BATCH_SIZE)	#	Train model over past data

		score_list.append(episodic_reward)
		avg_reward = np.mean(score_list[-100:])

		if avg_reward > best_score:
			best_score = avg_reward
			agent.save()

		print("%3d || Score: %.2f || Avg Score: %.2f || Best Score: %.2f || Frames: %3d || Eps: %.2f"\
			%(e+1, episodic_reward, avg_reward, best_score, env.frame_cnt, agent.eps))

		if (e+1) % EPS_STEP == 0:
			agent.reduceEps()

		if (e+1) % SAVE_AFTER:
			agent.save()
 
if __name__ == "__main__":
	main()