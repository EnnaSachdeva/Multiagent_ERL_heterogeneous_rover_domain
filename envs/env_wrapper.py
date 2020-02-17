import numpy as np, sys


class RoverDomainPython:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.rover_domain_python import RoverDomainVel

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = RoverDomainVel(args.config)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])
			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done); joint_global.append(info)

		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global

	def render(self):

		rand_univ = np.random.randint(0, len(self.universe))
		try: self.universe[rand_univ].render()
		except: 'Error rendering'

class RoverHeterogeneous:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.rover_heterogeneous import RoverDomainHeterogeneous

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = RoverDomainHeterogeneous(args.config)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])


			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done); joint_global.append(info)

			#if(len(next_state[0])!=111 or len(next_state[1])!=111 or len(next_state[2])!=111 or len(next_state[3])!=111):
			#	print("NO...................", len(next_state[0]), len(next_state[1]), len(next_state[2]), len(next_state[3]))
				#next_state, reward, done, info = env.step(action[:, universe_id, :])

		try: joint_obs = np.stack(joint_obs, axis=1)
		except:
			print("Not happening", (len(joint_obs), len(joint_obs[0])))

		joint_reward = np.stack(joint_reward, axis=1)


		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):

		rand_univ = np.random.randint(0, len(self.universe))
		try: self.universe[rand_univ].render()
		except: 'Error rendering'



	def viz(self):

		rand_univ = np.random.randint(0, len(self.universe))
		try: self.universe[rand_univ].viz(save=False, fname='path')
		except: 'Error rendering'



