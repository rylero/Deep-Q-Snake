import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
	def __init__(self):
		self.n_games = 0
		self.epsilon = 0 # randomness
		self.gamma = 0.9 # discount rate
		self.memory = deque(maxlen=MAX_MEMORY)
		self.model = Linear_QNet(11, 256, 3)
		self.record = 0
		if input("Use existing model (y/n)? ") == "y":
			# load model
			loaded_checkpoint = torch.load("./model/model.pth")
			print(loaded_checkpoint)
			self.n_games = loaded_checkpoint["epoch"]
			self.record = loaded_checkpoint["record"]
			self.model.load_state_dict(loaded_checkpoint["model_state"])
			self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
			self.trainer.optimizer.load_state_dict(loaded_checkpoint["optim_state"])
		else:
			self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
		
		if input("Save model PB's?").lower() in ["y", "yes"]:
			self.save_pb = True
		else:
			self.save_pb = False

	def get_state(self, game):
		head = game.snake[0]
		point_l = Point(head.x - 20, head.y)
		point_r = Point(head.x + 20, head.y)
		point_u = Point(head.x, head.y - 20)
		point_d = Point(head.x, head.y + 20)

		dir_l = game.direction == Direction.LEFT
		dir_r = game.direction == Direction.RIGHT
		dir_u = game.direction == Direction.UP
		dir_d = game.direction == Direction.DOWN

		state = [
			#Danger straight
			(dir_r and game.is_collision(point_r)) or
			(dir_l and game.is_collision(point_l)) or
			(dir_u and game.is_collision(point_u)) or
			(dir_d and game.is_collision(point_d)),

			#Danger right
			(dir_u and game.is_collision(point_r)) or
			(dir_d and game.is_collision(point_l)) or
			(dir_l and game.is_collision(point_u)) or
			(dir_r and game.is_collision(point_d)),

			#Danger left
			(dir_d and game.is_collision(point_r)) or
			(dir_u and game.is_collision(point_l)) or
			(dir_r and game.is_collision(point_u)) or
			(dir_l and game.is_collision(point_d)),

			# Move Direction
			dir_l,
			dir_r,
			dir_u,
			dir_d,

			# Food location
			game.food.x < game.head.x, #food is left
			game.food.x > game.head.x, #food is right
			game.food.y < game.head.y, #food is up
			game.food.y > game.head.y, #food is down
		]
		return np.array(state, dtype=int)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

	def train_long_memory(self):
		if len(self.memory) > BATCH_SIZE:
			mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
		else:
			mini_sample = self.memory
		states, actions, rewards, next_states, dones = zip(*mini_sample) # create seperate tuples for mini_sample
		self.trainer.train_step(states, actions, rewards, next_states, dones) # train the batch of data


	def train_short_memory(self, state, action, reward, next_state, done):
		self.trainer.train_step(state, action, reward, next_state, done) # train on one peice of data

	def get_action(self, state, game):
		# random moves: tradeoff exploration / explotation
		if not game.explore:
			self.epsilon = 0
		else:
			self.epsilon = np.clip((150 - self.n_games), 7, 120)
		final_move = [0,0,0]
		if random.randint(0, 200) < self.epsilon: # do a random move
			move = random.randint(0, 2)
			final_move[move] = 1
		else: # do models predicted move
			state0 = torch.tensor(state, dtype=torch.float)
			prediction = self.model(state0)
			move = torch.argmax(prediction).item()
			final_move[move] = 1
		return final_move

def train():
	plot_scores = []
	plot_mean_scores = []
	total_score = 0
	agent = Agent()
	game = SnakeGameAI()

	while True:
		# get current state
		state_current = agent.get_state(game)

		# get move
		final_move = agent.get_action(state_current, game)

		#perform move and get new state
		reward, done, score, save = game.play_step(final_move)
		state_new = agent.get_state(game)

		#train short memory
		agent.train_short_memory(state_current, final_move, reward, state_new, done)

		#remeber
		agent.remember(state_current, final_move, reward, state_new, done)
		
		if save:
			agent.model.save(agent.n_games, agent.model, agent.trainer.optimizer, agent.record)

		if done:
			# train long memory, plot result
			game.reset()
			agent.n_games +=1
			agent.train_long_memory()

			# check if we set a new high score
			if score > agent.record:
				agent.record = score
				if agent.save_pb:
					agent.model.save(agent.n_games, agent.model, agent.trainer.optimizer, agent.record)

			print("Game: ", agent.n_games, " Score: ", score, " Record: ", agent.record)
			
			plot_scores.append(score)
			total_score += score
			mean_score = total_score / agent.n_games
			plot_mean_scores.append(mean_score)
			plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
	train()