import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()
		# create layers
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		return x

	def save(self, number_of_games, model, optimizer, record, file_name='model.pth'):
		# get path for save
		model_folder_path = './model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)
		file_name = os.path.join(model_folder_path, file_name)
		checkpoint = {
			"epoch": number_of_games, # number of games played
			"record": record, # snakes record
			"model_state": model.state_dict(), # model state
			"optim_state": optimizer.state_dict() # optimizer state
		}
		torch.save(checkpoint, file_name)
		print("Saved")

class QTrainer:
	def __init__(self, model, lr=0.001, gamma=0.9):
		self.lr = lr
		self.gamma = gamma
		self.model = model
		self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss() # loss function

	def train_step(self, state, action, reward, next_state, done):
		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)

		if len(state.shape) == 1: # if training short memory instead of long (there will only be one value for short vs a batch size for long)
			# (1, x)
			state = torch.unsqueeze(state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			done = (done, )

		# 1: predicted Q values with current state
		pred = self.model(state)

		# 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
		# pred.clone()
		# preds[argmax(action)] = Q_new

		target = pred.clone()
		for idx in range(len(done)):
			Q_new = reward[idx]
			if not done[idx]:
				Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
			target[idx][torch.argmax(action).item()] = Q_new

		# optimize and compute loss
		self.optimizer.zero_grad()
		loss = self.criterion(target, pred)
		loss.backward()

		self.optimizer.step()