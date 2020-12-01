import gym
import sys
import time
import random
import utils_for_q_learning, buffer_class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

from RBFDQN import rbf_function, rbf_function_on_action, Net
from her_buffer import her_sampler, replay_buffer

STRAT='future'
K = 4
def get_env_params(env):
	obs = env.reset()
	# close the environment
	params = {'obs': obs['observation'].shape[0],
			'goal': obs['desired_goal'].shape[0],
			'action': env.action_space.shape[0],
			'action_max': env.action_space.high[0],
			}
	params['max_timesteps'] = env._max_episode_steps
	return params

class Net_HER(Net):
	def __init__(self, params, env, state_size, action_size, device):
		super(Net, self).__init__()

		self.env = env
		self.device = device
		self.params = params
		self.N = self.params['num_points']
		self.max_a = self.env.action_space.high[0]
		self.beta = self.params['temperature']

		env_params = get_env_params(self.env)
		self.her_module = her_sampler(STRAT, K, self.env.compute_reward)
		self.buffer_object = replay_buffer(env_params,
									self.params['max_buffer_size'],
									self.her_module.sample_her_transitions)

		self.state_size, self.action_size = state_size, action_size
		self.input_size = env_params['obs'] + env_params['goal']

		self.value_module = nn.Sequential(
			nn.Linear(self.input_size, self.params['layer_size']),
			nn.ReLU(),
			nn.Linear(self.params['layer_size'], self.params['layer_size']),
			nn.ReLU(),
			nn.Linear(self.params['layer_size'], self.params['layer_size']),
			nn.ReLU(),
			nn.Linear(self.params['layer_size'], self.N),
		)

		if self.params['num_layers_action_side'] == 1:
			self.location_module = nn.Sequential(
				nn.Linear(self.input_size, self.params['layer_size_action_side']),
				nn.Dropout(p=self.params['dropout_rate']),
				nn.ReLU(),
				nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
				utils_for_q_learning.Reshape(-1, self.N, self.action_size),
				nn.Tanh(),
			)
		elif self.params['num_layers_action_side'] == 2:
			self.location_module = nn.Sequential(
				nn.Linear(self.input_size, self.params['layer_size_action_side']),
				nn.Dropout(p=self.params['dropout_rate']),
				nn.ReLU(),
				nn.Linear(self.params['layer_size_action_side'], self.params['layer_size_action_side']),
				nn.Dropout(p=self.params['dropout_rate']),
				nn.ReLU(),
				nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
				utils_for_q_learning.Reshape(-1, self.N, self.action_size),
				nn.Tanh(),
			)

		torch.nn.init.xavier_uniform_(self.location_module[0].weight)
		torch.nn.init.zeros_(self.location_module[0].bias)

		self.location_module[3].weight.data.uniform_(-.1, .1)
		self.location_module[3].bias.data.uniform_(-1., 1.)

		self.criterion = nn.MSELoss()

		self.params_dic = [{
			'params': self.value_module.parameters(), 'lr': self.params['learning_rate']
		},
						   {
							   'params': self.location_module.parameters(),
							   'lr': self.params['learning_rate_location_side']
						   }]
		try:
			if self.params['optimizer'] == 'RMSprop':
				self.optimizer = optim.RMSprop(self.params_dic)
			elif self.params['optimizer'] == 'Adam':
				self.optimizer = optim.Adam(self.params_dic)
			else:
				print('unknown optimizer ....')
		except:
			print("no optimizer specified ... ")


		self.to(self.device)

	def get_centroid_values(self, s):
		'''
		given a batch of s, get all centroid values, [batch x N]
		'''
		centroid_values = self.value_module(s)
		return centroid_values

	def get_centroid_locations(self, s):
		'''
		given a batch of s, get all centroid_locations, [batch x N x a_dim]
		'''
		centroid_locations = self.max_a * self.location_module(s)
		return centroid_locations

	def get_best_qvalue_and_action(self, s):
		'''
		given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
		'''
		all_centroids = self.get_centroid_locations(s)
		values = self.get_centroid_values(s)
		weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]
		allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids
		# a -> all_centroids[idx] such that idx is max(dim=1) in allq
		# a = torch.gather(all_centroids, dim=1, index=indices)
		# (dim: bs x 1, dim: bs x action_dim)
		best, indices = allq.max(dim=1)
		if s.shape[0] == 1:
			index_star = indices.item()
			a = all_centroids[0, index_star]
			return best, a
		else:
			return best, None

	def forward(self, s, a):
		'''
		given a batch of s,a , compute Q(s,a) [batch x 1]
		'''
		centroid_values = self.get_centroid_values(s)  # [batch_dim x N]
		centroid_locations = self.get_centroid_locations(s)
		# [batch x N]
		centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
		output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
		output = output.sum(1, keepdim=True)  # [batch x 1]
		return output

	def e_greedy_policy(self, s, episode, train_or_test):
		'''
		Given state s, at episode, take random action with p=eps if training
		Note - epsilon is determined by episode
		'''
		epsilon = 1.0 / np.power(episode, 1.0 / self.params['policy_parameter'])
		if train_or_test == 'train' and random.random() < epsilon:
			a = self.env.action_space.sample()
			return a.tolist()
		else:
			self.eval()
			s_matrix = np.array(s).reshape(1, self.input_size)
			with torch.no_grad():
				s = torch.from_numpy(s_matrix).float().to(self.device)
				_, a = self.get_best_qvalue_and_action(s)
				a = a.cpu().numpy()
			self.train()
			return a

	def e_greedy_gaussian_policy(self, s, episode, train_or_test):
		'''
		Given state s, at episode, take random action with p=eps if training
		Note - epsilon is determined by episode
		'''
		epsilon = 1.0 / np.power(episode, 1.0 / self.params['policy_parameter'])
		if train_or_test == 'train' and random.random() < epsilon:
			a = self.env.action_space.sample()
			return a.tolist()
		else:
			self.eval()
			s_matrix = np.array(s).reshape(1, self.input_size)
			with torch.no_grad():
				s = torch.from_numpy(s_matrix).float().to(self.device)
				_, a = self.get_best_qvalue_and_action(s)
				a = a.cpu().numpy()
			self.train()
			noise = np.random.normal(loc=0.0, scale=self.params['noise'], size=len(a))
			a = a + noise
			return a

	def gaussian_policy(self, s, episode, train_or_test):
		'''
		Given state s, at episode, take random action with p=eps if training
		Note - epsilon is determined by episode
		'''
		self.eval()
		s_matrix = np.array(s).reshape(1, self.input_size)
		with torch.no_grad():
			s = torch.from_numpy(s_matrix).float().to(self.device)
			_, a = self.get_best_qvalue_and_action(s)
			a = a.cpu()
		self.train()
		noise = np.random.normal(loc=0.0, scale=self.params['noise'], size=len(a))
		a = a + noise
		return a


	def update(self, target_Q, count):
		"""
		remaining question: if agent achieved goal, is that a "done"?
		i.e, should done be 1 if s==g?

		at present, there are no dones!
		"""
		if len(self.buffer_object) < self.params['batch_size']:
			return 0

		transitions = self.buffer_object.sample(self.params['batch_size'])
		s_matrix = transitions['obs']
		sp_matrix = transitions['obs_next']
		g_matrix =  transitions['g']
		a_matrix = transitions['actions']
		done_matrix = np.zeros(len(s_matrix)).reshape(-1,1)

		inputs = np.concatenate([s_matrix, g_matrix], axis=1)
		inputs_prime = np.concatenate([sp_matrix, g_matrix], axis=1)

		r_matrix = transitions['r']
		r_matrix = np.clip(r_matrix,
							  a_min=-self.params['reward_clip'],
							  a_max=self.params['reward_clip'])

		s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
		a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
		r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
		done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
		sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

		inputs_tensor = torch.from_numpy(inputs).float().to(self.device)
		inputs_prime_tensor = torch.from_numpy(inputs_prime).float().to(self.device)

		Q_star, _ = target_Q.get_best_qvalue_and_action(inputs_prime_tensor)
		Q_star = Q_star.reshape((self.params['batch_size'], -1))
		with torch.no_grad():
			y = r_matrix + self.params['gamma'] * (1 - done_matrix) * Q_star

		y_hat = self.forward(inputs_tensor, a_matrix)

		loss = self.criterion(y_hat, y)
		self.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.zero_grad()
		utils_for_q_learning.sync_networks(
			target=target_Q,
			online=self,
			alpha=self.params['target_network_learning_rate'],
			copy=False)
		return loss.cpu().data.numpy()


if __name__ == '__main__':
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print("Running on the GPU")
	else:
		device = torch.device("cpu")
		print("Running on the CPU")

	sys.stdout.flush()
	hyper_parameter_name = sys.argv[1]
	alg = 'rbf'
	params = utils_for_q_learning.get_hyper_parameters(hyper_parameter_name, alg)
	params['hyper_parameters_name'] = hyper_parameter_name
	env = gym.make(params['env_name'])
	#env = gym.wrappers.Monitor(env, 'videos/'+params['env_name']+"/", video_callable=lambda episode_id: episode_id%10==0,force = True)
	params['env'] = env
	params['seed_number'] = int(sys.argv[2])
	utils_for_q_learning.set_random_seed(params)
	s0 = env.reset()
	utils_for_q_learning.action_checker(env)
	Q_object = Net_HER(params,
				   env,
				   state_size=len(s0),
				   action_size=len(env.action_space.low),
				   device=device)
	Q_object_target = Net_HER(params,
						  env,
						  state_size=len(s0),
						  action_size=len(env.action_space.low),
						  device=device)
	Q_object_target.eval()

	utils_for_q_learning.sync_networks(target=Q_object_target,
									   online=Q_object,
									   alpha=params['target_network_learning_rate'],
									   copy=True)

	G_li = []
	loss_li = []
	all_times_per_steps = []
	all_times_per_updates = []
	for episode in range(params['max_episode']):
		sys.stdout.flush()
		print("episode {}".format(episode))
		Q_this_episode = Net_HER(params,
							 env,
							 state_size=len(s0),
							 action_size=len(env.action_space.low),
							 device=device)
		utils_for_q_learning.sync_networks(target=Q_this_episode,
										   online=Q_object,
										   alpha=params['target_network_learning_rate'],
										   copy=True)
		Q_this_episode.eval()

		ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
		observation, done, t = env.reset(), False, 0

		s = observation['observation']
		ag = observation['achieved_goal']
		g = observation['desired_goal']
		while not done:
			input=np.concatenate([s, g])
			if params['policy_type'] == 'e_greedy':
				a = Q_object.e_greedy_policy(input, episode + 1, 'train')
			elif params['policy_type'] == 'e_greedy_gaussian':
				a = Q_object.e_greedy_gaussian_policy(input, episode + 1, 'train')
			elif params['policy_type'] == 'gaussian':
				a = Q_object.gaussian_policy(input, episode + 1, 'train')

			a = np.array(a)
			obs_new, r, done, _ = env.step(a)
			sp = obs_new['observation']
			ag_new = obs_new['achieved_goal']
			t = t + 1
			done_p = False if t == env._max_episode_steps else done

			ep_obs.append(s.copy())
			ep_ag.append(ag.copy())
			ep_g.append(g.copy())
			ep_actions.append(a.copy())

			s = sp
			ag = ag_new

		ep_obs.append(s.copy())
		ep_ag.append(ag.copy())

		mb_obs = np.array([ep_obs])
		mb_ag = np.array([ep_ag])
		mb_g = np.array([ep_g])
		mb_actions = np.array([ep_actions])
		Q_object.buffer_object.store_episode([mb_obs, mb_ag, mb_g, mb_actions])

		#now update the Q network
		loss = []
		for count in range(params['updates_per_episode']):
			temp = Q_object.update(Q_object_target, count)
			loss.append(temp)

		loss_li.append(np.mean(loss))

		if (episode % 10 == 0) or (episode == params['max_episode'] - 1):
			temp = []
			# TODO: success rate in eval loop below
			for _ in range(10):
				observation, G, done, t = env.reset(), 0, False, 0
				s = observation['observation']
				ag = observation['achieved_goal']
				g = observation['desired_goal']
				while done == False:
					input = np.concatenate([s,g])
					a = Q_object.e_greedy_policy(input, episode + 1, 'test')
					new_observation, r, done, _ = env.step(np.array(a))
					s, G, t = new_observation['observation'], G + r, t + 1
				temp.append(G)
			print(
				"after {} episodes, learned policy collects {} average returns".format(
					episode, np.mean(temp)))
			G_li.append(np.mean(temp))
			utils_for_q_learning.save(G_li, loss_li, params, alg)
