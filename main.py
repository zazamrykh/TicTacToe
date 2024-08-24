import copy
import time
from enum import Enum
import pickle
import numpy as np

from state_action_calculation import check_winning_condition, calculate_states_and_actions, visualise_state, \
    get_equivalent, get_initial


class PlayerType(Enum):
    CROSS = 1,
    CIRCLE = 2


class TicTacToeEnv:
    def __init__(self, observations, env_player):
        self.observations = observations
        self.init_observation = [0 for _ in range(9)]
        self.observation = copy.deepcopy(self.init_observation)
        self.env_player = env_player
        self.step_i = 0
        self.max_step_n = 9

    def reset(self):
        if self.env_player.player_type == PlayerType.CIRCLE:
            self.observation = self.init_observation.copy()
            self.step_i = 0
        else:
            self.observation = self.init_observation.copy()
            _, action = self.env_player.get_action(self.observation)
            self.observation[action] = self.env_player.player_type.value[0]
            self.step_i = 1
        return self.observation

    def step(self, action, player_type=PlayerType.CROSS):
        new_observation = self.observation

        if new_observation[action] != 0:
            raise Exception("Wrong action!")

        if player_type == PlayerType.CROSS:
            new_observation[action] = player_type.value[0]  # our step
            self.step_i += 1

            if check_winning_condition(new_observation):
                reward = 1
                end = True
                return new_observation, reward, end

            if self.step_i >= self.max_step_n:  # draw
                reward = 0
                end = True
                return new_observation, reward, end

            _, env_player_action = self.env_player.get_action(new_observation)

            new_observation[env_player_action] = self.env_player.player_type.value
            self.step_i += 1

            if check_winning_condition(new_observation):
                reward = -1
                end = True
                return new_observation, reward, end

            if self.step_i >= self.max_step_n:  # draw
                reward = 0
                end = True
                return new_observation, reward, end

            reward = 0  # game still on
            end = False
            return new_observation, reward, end
        else:
            new_observation[action] = player_type.value  # our step
            self.step_i += 1

            if check_winning_condition(new_observation):
                reward = 1
                end = True
                return new_observation, reward, end

            if self.step_i >= self.max_step_n:  # draw
                reward = 0
                end = True
                return new_observation, reward, end

            _, env_player_action = self.env_player.get_action(new_observation)

            new_observation[env_player_action] = self.env_player.player_type.value[0]
            self.step_i += 1

            if check_winning_condition(new_observation):
                reward = -1
                end = True
                return new_observation, reward, end

            if self.step_i >= self.max_step_n:  # draw
                reward = 0
                end = True
                return new_observation, reward, end

            reward = 0  # game still on
            end = False
            return new_observation, reward, end

    def render(self):
        visualise_state(self.observation)


def observation_to_state(state):
    result = 0
    for i, n in enumerate(state):
        result += n * 3 ** (8 - i)
    return result


def state_to_observation(num):
    result = []
    for i in range(9):
        result.append(num // 3 ** (8 - i))
        num %= 3 ** (8 - i)
    return result


class TicTacToeAgent():
    def __init__(self, states, actions, player_type=PlayerType.CROSS):
        self.model = {observation_to_state(state): (action, [1 / len(action) for _ in range(len(action))])
                      for state, action in zip(states, actions)}  # state : availeble_act, propas
        self.player_type = player_type

    def get_action(self, observation):
        if type(observation) is not list:
            observation = state_to_observation(observation)
        equivalent, translation = get_equivalent(observations, observation)

        state = observation_to_state(equivalent)
        available_actions = self.model[state][0]
        probabilities = self.model[state][1]
        action = np.random.choice(available_actions, p=probabilities / np.sum(probabilities)).item()
        return action, get_initial(action, translation)  # equivalent and initial

    def save_model(self, file_name):
        if '.pkl' not in file_name:
            file_name += ".pkl"
        with open(file_name, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, file_name):
        if '.pkl' not in file_name:
            file_name += ".pkl"
        with open(file_name, 'rb') as f:
            self.model = pickle.load(f)


class CrossEntropyAgent(TicTacToeAgent):
    def fit(self, elite_trajectories, alpha=0.0, lamb=1.0):
        new_model = {observation_to_state(obs): (action, [0 for _ in range(len(action))])
                     for obs, action in zip(observations, available_actions)}
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                index_of_action = new_model[state][0].index(action)
                new_model[state][1][index_of_action] += 1

        for state in new_model:
            actions_state_counts = new_model[state][1]
            state_count = np.sum(actions_state_counts)
            actions_num = len(actions_state_counts)
            if state_count > 0:
                if alpha != 0.0:
                    for i in range(actions_num):
                        actions_state_counts[i] += alpha
                actions_state_counts /= (state_count + alpha * actions_num)

                if lamb != 1.0:
                    for i in range(actions_num):
                        actions_state_counts[i] = lamb * actions_state_counts[i] + (1 - lamb) * self.model[state][1][i]
                for i, prob in enumerate(self.model[state][1]):
                    new_model[state][1][i] = actions_state_counts[i].item()
            else:
                for i, prob in enumerate(self.model[state][1]):
                    new_model[state][1][i] = prob

        self.model = new_model
        return None


def get_trajectory(env, agent, max_len=9, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    observation = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(observation_to_state(get_equivalent(observations, observation)[0]))

        equivalent_act, init_act = agent.get_action(observation)
        trajectory['actions'].append(equivalent_act)

        observation, reward, done = env.step(init_act, player_type=agent.player_type)
        trajectory['rewards'].append(reward)

        if visualize:
            env.render()

        if done:
            break

    return trajectory


def train_ce(env, player, iteration_n, trajectory_n, q_param):
    for iteration in range(iteration_n):

        # policy evaluation
        visualize = False
        if iteration % 25 == 0:
            visualize = True
        trajectories = []
        for _ in range(trajectory_n):
            trajectory = get_trajectory(env, player, visualize=visualize)
            visualize = False
            trajectories.append(trajectory)

        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward >= quantile:
                elite_trajectories.append(trajectory)

        player.fit(elite_trajectories, lamb=0.85)


observations, available_actions = calculate_states_and_actions()


# After training both cross and circle players union them
def union_models(cross_model, circle_model, united_model):
    model = united_model
    for state in model:
        if len(model[state][0]) % 2 == 1:  # cross step
            for i, prob in enumerate(cross_model[state][1]):
                model[state][1][i] = prob
        else:
            for i, prob in enumerate(circle_model[state][1]):
                model[state][1][i] = prob


def train_cross(q_param, iteration_n, trajectory_n):
    env_player = CrossEntropyAgent(observations, available_actions, player_type=PlayerType.CIRCLE)
    env_player.load_model("CE_player_vs_random.pkl")
    env = TicTacToeEnv(observations, env_player)
    env.reset()

    player = CrossEntropyAgent(observations, available_actions, player_type=PlayerType.CROSS)
    player.load_model("CE_player_vs_random.pkl")
    train_ce(env, player, iteration_n, trajectory_n, q_param)

    trajectory = get_trajectory(env, player, visualize=True)
    print('total reward:', sum(trajectory['rewards']))
    print('model:')
    print(player.model)
    player.save_model("CE_player_cross_vs_CE_player.pkl")


def train_circles(q_param, iteration_n, trajectory_n):
    env_player = CrossEntropyAgent(observations, available_actions, player_type=PlayerType.CROSS)
    env_player.load_model("CE_player_vs_random.pkl")

    env = TicTacToeEnv(observations, env_player)

    env.reset()

    player = CrossEntropyAgent(observations, available_actions, player_type=PlayerType.CIRCLE)
    player.load_model("CE_player_vs_random.pkl")
    train_ce(env, player, iteration_n, trajectory_n, q_param)

    trajectory = get_trajectory(env, player, visualize=True)
    print('total reward:', sum(trajectory['rewards']))
    print('model:')
    print(player.model)
    player.save_model("CE_player_circle_vs_CE_player.pkl")


def union_agents():
    cross_player = CrossEntropyAgent(observations, available_actions, player_type=PlayerType.CROSS)
    cross_player.load_model("CE_player_cross_vs_CE_player.pkl")
    circle_player = CrossEntropyAgent(observations, available_actions, player_type=PlayerType.CIRCLE)
    circle_player.load_model("CE_player_circle_vs_CE_player.pkl")
    united_player = CrossEntropyAgent(observations, available_actions)

    union_models(cross_player.model, circle_player.model, united_player.model)
    united_player.save_model("CE_player_vs_CE.pkl")


def play_game(bot_ver, player_type):
    bot_type = PlayerType.CROSS if player_type == PlayerType.CIRCLE else PlayerType.CIRCLE

    bot = CrossEntropyAgent(observations, available_actions, player_type=bot_type)
    bot.load_model(bot_ver)

    env = TicTacToeEnv(observations, bot)
    while True:
        observation = env.reset()

        max_len = 9
        result_reward = 0
        for _ in range(max_len):
            env.render()
            action = int(input("Input action"))
            while observation[action] != 0:
                action = int(input("Input action"))
            observation, reward, done = env.step(action, player_type=player_type)
            result_reward = reward
            if done:
                break

        print(result_reward)

if __name__ == "__main__":
    play_game('CE_player_vs_CE.pkl', PlayerType.CIRCLE)
    # q_param = 0.6
    # iteration_n = 100
    # trajectory_n = 1500
    # print('Train crosses')
    # train_cross(q_param, iteration_n, trajectory_n)
    #
    # print('Train circles')
    # train_circles(q_param, iteration_n, trajectory_n)
    #
    # union_agents()
