import random
import pathlib
import pickle

import crafter
import gymnasium
import numpy as np
from crafter import Env as CrafterEnv
from crafter import constants as crafter_constants

import gym
from gym import spaces

import embodied
from . import from_gym

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


class PatchedCrafterEnv(embodied.Env):
    achievements_list = [
        'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone',
        'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
        'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword',
        'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up',
    ]

    def __init__(self, config_task=None, recorder_dir=None, vis=False):
        self._env = CrafterEnv()
        if recorder_dir:
            # VideoRecorder should be fixed, to work with this patched environment
            crafter.Recorder(self._env, recorder_dir, save_video=True)
        self.num_agents = 1
        self._size = (64, 64)
        self._task_vector_size = 32
        self._subtask_extra_reward = 1.0

        self._current_achievement_task = None
        self._previous_achievement_count = None
        self._tasks = None
        self._step = None

        self._current_achievement_tasks = None

        self.wrappers = [
          from_gym.FromGym,
          lambda e: embodied.wrappers.ResizeImage(e, self._size),
        ]

        # Adding embeddings
        self.dataset_type, self.mode, enc = config_task.split('_')

        directory = pathlib.Path(__file__).resolve().parent

        if self.mode == 'train':
            if enc == 'old':
                with open(directory / "data/dataset_embeds.pkl", "rb") as f:
                    self.cache, self.embed_cache, self.LLM_discription_of_medium_instuctions = pickle.load(f)
            else:
                with open(directory / "data/dataset_embeds_train_new_enc.pkl", "rb") as f:
                    self.cache, self.embed_cache, self.LLM_discription_of_medium_instuctions = pickle.load(f)
        else:
            self.id_task = 0
            self.test_info = dict()

            if enc == 'old':
                with open(directory / "data/dataset_embeds_test.pkl", "rb") as f:
                    self.cache, self.embed_cache, self.LLM_discription_of_medium_instuctions = pickle.load(f)
            else:
                with open(directory / "data/dataset_embeds_test_new_enc.pkl", "rb") as f:
                    self.cache, self.embed_cache, self.LLM_discription_of_medium_instuctions = pickle.load(f)
                
            self.name_test_info = 'test_data/info_' + config_task + '.pkl'

        
        if self.dataset_type == 'MediumInstructions':
            self._tasks = list(self.LLM_discription_of_medium_instuctions.values())
        elif self.dataset_type == 'HardInstructions':
            self._tasks = list(self.LLM_discription_of_medium_instuctions.keys())
        elif self.dataset_type == 'MixedMediumHardInstructions':
            self._tasks = list(self.LLM_discription_of_medium_instuctions.values()) + list(self.LLM_discription_of_medium_instuctions.keys())
        else:
            assert self.dataset_type == 'random'

        self.len_test = len(self._tasks)
        print("Len tasks", self.len_test)

        self.empty_token = self.cache["<pad>"]
        self.tokens = [self.empty_token]
        self.cur_token = 0
        self.embed_size = 512

        # For testing mode
        self.vis = vis
        self.prev_action = None

        # Add metrics
        self._achievements = crafter.constants.achievements.copy()


    def render(self):
        self._env.render()
    
    def render_with_text(self, text):
        img = self._env.render()
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, (0, 0, 0))
        draw.text((0, 45), "Action: {}".format(self.prev_action), (0, 0, 0))
        img = np.asarray(img)
        return img

    @property
    def observation_space(self):

        image = spaces.Box(low=0, high=1, shape=self._size)
        log_language_info = spaces.Text(max_length=10000)
        token = spaces.Box(0, 32100, shape=(), dtype=np.uint32)
        token_embed = spaces.Box(-np.inf, np.inf, shape=(self.embed_size,), dtype=np.float32)
        is_read_step = spaces.Box(low=np.array(False), high=np.array(True), shape=(), dtype=bool)

        obs_space = spaces.Dict({
            'image': image,
            'log_language_info': log_language_info,
            'token': token,
            'token_embed': token_embed,
            "is_read_step": is_read_step,
        })
        
        if self.vis:
            obs_space["log_image"] = spaces.Box(
                low=0,
                high=255,
                shape=(100 * self._size[0], 100 * self._size[1], 3),
            )

        # Add metrics
        
        for k in self._achievements:
            obs_space[f'log_achievement_{k}'] = spaces.Box(-1, 100, shape=(), dtype=np.uint32)

        obs_space['log_achievement_TaskScore'] = spaces.Box(-1, 100, shape=(), dtype=np.uint32)
        obs_space['log_achievement_TaskStepsToSuccess'] = spaces.Box(-1, 1000, shape=(), dtype=np.uint32)
        obs_space['log_achievement_SuccessRate'] = spaces.Box(0, 1, shape=(), dtype=np.uint32)

        return obs_space

    @property
    def action_space(self):
        # noinspection PyUnresolvedReferences
        return gymnasium.spaces.Discrete(len(crafter_constants.actions))

    def _get_task_vector(self):
        # Create a one-hot encoded vector for the current achievement task
        task_vector = np.zeros(self._task_vector_size, dtype=np.float32)
        if self._current_achievement_task:
            task_index = self.achievements_list.index(self._current_achievement_task)
            task_vector[task_index] = 1
        return task_vector
    
    def _get_and_set_task_embed(self):
        if self._current_achievement_task:
            if isinstance(self._current_achievement_task, str):
                self.string = ' '.join(self._current_achievement_task.split('_'))
                self.tokens = [x for x in self.cache[self.string]]
                self.token_embeds = [x for x in self.embed_cache[self.string]]
            else:
                strings = [' '.join(x.split('_')) for x in self._current_achievement_task]
                self.string = ' '.join(strings)
                self.tokens = [x for string in strings for x in self.cache[string]]
                self.token_embeds = [x for string in strings for x in self.embed_cache[string]]
        else:
            self.string = ''
            self.tokens = [self.empty_token]
            self.token_embeds = [self.embed_cache["<pad>"]]
        self.cur_token = 0

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        reward /= 10
        self._step += 1

        terminated = info['discount'] < 1
        truncated = done if not terminated else False

        # Code to handle additional reward for the current achievement task
        if self._current_achievement_task:
            if self.dataset_type == 'MediumInstructions':
                for achievement in self._current_achievement_task:
                    cur_achievement_count = info['achievements'].get(achievement, 0)

                    if cur_achievement_count > self._previous_achievement_count[achievement]:
                        reward += self._subtask_extra_reward
                        self._previous_achievement_count[achievement] = cur_achievement_count

                        self._current_achievement_task.remove(achievement)
                        if len(self._current_achievement_task) > 0:
                            self._get_and_set_task_embed()
                            self.cur_token = -1
                        else:
                            terminated = True
                        break

            elif  self.dataset_type == 'HardInstructions' or self.dataset_type == 'MixedMediumHardInstructions':
                for achievement in self._current_achievement_tasks:
                    cur_achievement_count = info['achievements'].get(achievement, 0)

                    if cur_achievement_count > self._previous_achievement_count[achievement]:
                        # print(achievement)
                        reward += self._subtask_extra_reward
                        self._previous_achievement_count[achievement] = cur_achievement_count

                        self._current_achievement_tasks.remove(achievement)
                        if len(self._current_achievement_tasks) == 0:
                            terminated = True
                        break

            else:
                cur_achievement_count = info['achievements'].get(self._current_achievement_task, 0)

                if cur_achievement_count > self._previous_achievement_count:
                    reward += self._subtask_extra_reward
                    self._previous_achievement_count = cur_achievement_count

                    self._current_achievement_tasks.pop(0)
                    if len(self._current_achievement_tasks) > 0:
                        self._current_achievement_task = self._current_achievement_tasks[0]
                        self._get_and_set_task_embed()
                        self.cur_token = -1
                        self._previous_achievement_count = 0
                    else:
                        terminated = True

        self.cur_token += 1
        if self.cur_token >= len(self.tokens):
            self.cur_token = 0

        augmented_obs = {
            'image': obs,
            'log_language_info': self.string,
            'token': self.tokens[self.cur_token],
            'token_embed': self.token_embeds[self.cur_token],
            "is_read_step": False, 
        }

        done = terminated or truncated

        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        if done:
            self._update_episode_stats(info)

        # Add metrics
            for key, value in info['episode_extra_stats'].items():
                augmented_obs[key] = value
                # print(key, value)
            
            if self.mode == 'test':
                self.test_info[augmented_obs['log_language_info']] = self.test_info.get(augmented_obs['log_language_info'], []) + [(augmented_obs['log_achievement_TaskScore'], augmented_obs['log_achievement_SuccessRate'])]

                with open(self.name_test_info, 'wb') as f:
                    pickle.dump(self.test_info, f)
                
                if self.len_test == self.id_task:
                    self.id_task = 0
                # print(self.id_task)

        # End add metrics

        #if reward >= 0.7:
            # print(self._step, self.string, self.tokens[self.cur_token], self._current_achievement_tasks, reward)

        if self.vis:
            self.prev_action = self._env.action_names[action]
            augmented_obs["log_image"] = self.render_with_text(augmented_obs["log_language_info"])

        return augmented_obs, reward, done, info

    @staticmethod
    def _compute_scores(percents):
        scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
        return scores

    def _update_episode_stats(self, info, key='episode_extra_stats'):

        if self.dataset_type == 'MediumInstructions':
            for achievement in set(self._current_achievement_tasks):
                key_name = f'log_achievement_{achievement}'
                info[key][key_name] = self._previous_achievement_count[achievement]
            sum_previous_achievement_count = sum(self._previous_achievement_count.values())
            info[key]['log_achievement_TaskScore'] = sum_previous_achievement_count
            if sum_previous_achievement_count:
                info[key]['log_achievement_TaskStepsToSuccess'] = self._step
            if len(self._current_achievement_task) == 0:
                info[key]['log_achievement_SuccessRate'] = 1
            else:
                info[key]['log_achievement_SuccessRate'] = 0

        elif self.dataset_type == 'HardInstructions' or self.dataset_type == 'MixedMediumHardInstructions':
            # print(self.LLM_discription_of_medium_instuctions[self._current_achievement_task])
            if isinstance(self._current_achievement_task, str):
                achievements = set(self.LLM_discription_of_medium_instuctions[self._current_achievement_task])
            else:
                achievements = self._current_achievement_task
            for achievement in achievements:
                key_name = f'log_achievement_{achievement}'
                info[key][key_name] = self._previous_achievement_count[achievement]
            sum_previous_achievement_count = sum(self._previous_achievement_count.values())
            info[key]['log_achievement_TaskScore'] = sum_previous_achievement_count
            if sum_previous_achievement_count:
                info[key]['log_achievement_TaskStepsToSuccess'] = self._step
            if len(self._current_achievement_tasks) == 0:
                info[key]['log_achievement_SuccessRate'] = 1
            else:
                info[key]['log_achievement_SuccessRate'] = 0

        else:

            if self._current_achievement_task:
                key_name = f'log_achievement_{self._current_achievement_task}'
                info[key][key_name] = self._previous_achievement_count
                info[key]['log_achievement_TaskScore'] = self._previous_achievement_count
                if self._previous_achievement_count:
                    info[key]['log_achievement_TaskStepsToSuccess'] = self._step
            else:
                achievements = {'Achievements/' + ach: 100.0 if val > 0.0 else 0.0 for ach, val in
                                info['achievements'].items()}
                info[key] = achievements
                info[key]['Num_achievements'] = sum(val > 0 for val in achievements.values())
                info[key]['Score'] = self._compute_scores(np.array(list(achievements.values())))

    def reset(self, *args, **kwargs):

        if self.dataset_type == 'random':
            number_tasks = random.choice(range(6))
        elif self.dataset_type != 'MediumInstructions' and self.dataset_type != 'HardInstructions' and self.dataset_type != 'MixedMediumHardInstructions':
            number_tasks = int(self.dataset_type)
        
        if self.dataset_type == 'MediumInstructions':
            if self.mode == 'train':
                self._current_achievement_tasks = random.choice(self._tasks)
            else:
                self._current_achievement_task = self._tasks[self.id_task]
                self.id_task += 1

            self._current_achievement_task = self._current_achievement_tasks.copy() 
            self._previous_achievement_count = {}
            for task in set(self._current_achievement_tasks):
                self._previous_achievement_count[task] = 0
        
        elif self.dataset_type == 'HardInstructions' or self.dataset_type == 'MixedMediumHardInstructions':
            if self.mode == 'train':
                self._current_achievement_task = random.choice(self._tasks)
            else:
                self._current_achievement_task = self._tasks[self.id_task]
                self.id_task += 1

            if isinstance(self._current_achievement_task, str):
                self._current_achievement_tasks = self.LLM_discription_of_medium_instuctions[self._current_achievement_task].copy()
            else:
                self._current_achievement_tasks = self._current_achievement_task.copy()
            self._previous_achievement_count = {}
            for task in set(self._current_achievement_tasks):
                self._previous_achievement_count[task] = 0

        else:
            tasks = ['collect_coal', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone',
                        'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'make_iron_pickaxe',
                        'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe',
                        'make_wood_sword', 'wake_up'] + [None] * 5
            if number_tasks > 0:
                self._current_achievement_tasks = list(np.random.choice(tasks, size=number_tasks))
                self._current_achievement_task = self._current_achievement_tasks[0]
            else:
                self._current_achievement_tasks = []
                self._current_achievement_task = None
            self._previous_achievement_count = 0

        self._get_and_set_task_embed()
        self._step = 0

        original_obs = self._env.reset()
        augmented_obs = {
            'image': original_obs,
            'log_language_info': self.string,
            'token': self.tokens[self.cur_token],
            'token_embed': self.token_embeds[self.cur_token],
            "is_read_step": False
        }
        # print(self.string, self._current_achievement_tasks)

        if self.vis:
            self.prev_action = ""
            augmented_obs["log_image"] = self.render_with_text(augmented_obs["log_language_info"])

        return augmented_obs