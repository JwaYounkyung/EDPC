# sys.path = ['..'] + sys.path
# sys.path = ['../../mpe_local'] + sys.path
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from maddpg_o.maddpg_local.micro.maddpg import MADDPGAgentMicroSharedTrainer
from maddpg_o.maddpg_local.micro.rmaddpg import RMADDPGAgentMicroSharedTrainer
import maddpg_o.maddpg_local.common.tf_util as U
from model_v3_test3 import mlp_model_agent_p, mlp_model_adv_p, mlp_model_agent_q, mlp_model_adv_q, mlp_model, mean_field_adv_q_model, mean_field_agent_q_model
from model_v3_numbered import mlp_model_agent_p_numbered, mlp_model_adv_p_numbered, mlp_model_agent_q_numbered, mlp_model_adv_q_numbered
import argparse
import time
import re
import multiprocessing, logging
from functools import partial
from maddpg_o.experiments.train_helper.union_replay_buffer import UnionReplayBuffer
import numpy as np
import imageio
import queue
import joblib
import tempfile
import random
import gc

import pickle
# logger = multiprocessing.log_to_stderr()
# logger.setLevel(multiprocessing.SUBDEBUG)

FLAGS = None

# import multiagent
# print(multiagent.__file__)
# print(Scenario)

# ray.init()

# load model
# N_GOOD, N_ADV, N_LAND should align with the environment
N_GOOD = None
N_ADV = None
# N_LAND = num_landmarks+num_food+num_forests
N_LANDMARKS = None
N_FOOD = None
N_FORESTS = None
N_LAND = None

ID_MAPPING = None

INIT_WEIGHTS = None
GOOD_SHARE_WEIGHTS = False
ADV_SHARE_WEIGHTS = False
SHARE_WEIGHTS = None
CACHED_WEIGHTS = {}

WEIGHT_STACK = False

GRAPHS = []
SESSIONS = []
TRAINERS = []

CLUSTER = None
SERVERS = None

PERTURBATION = None
NEW_IDS = []


def exp_to_bytes(f, all_data):
    flat_data = []
    for data_n in all_data:
        if type(data_n) == list:
            for data in data_n:
                flat_data.append(data.flatten())
        else:
            flat_data.append(data_n.flatten())
    total_length = 0
    # print(len(flat_data))

    flat_data = np.concatenate(flat_data).astype(np.float32)
    b = flat_data.tobytes()
    # f.write(b)
    total_length = len(b)
    # s = 1
    # while s < total_length:
    #     s *= 2
    # b += b'f' * (s - total_length)
    # total_length = s
    # assert len(b) == s
    f.write(b)
    # print("tt", total_numbers)
    # f.flush()
    return total_length, flat_data


def bytes_to_exp(f, n):
    flat_data = np.fromfile(f, dtype=np.float32, count=n)
    # all_data = []
    # k = 0
    # for i in range(5):
    #     data_n = []
    #     for j in range(n):
    #         data_n.append(flat_data[k])
    #         k += 1
    #     all_data.append(data_n)
    return flat_data


def format_time(ti):
    h, m = divmod(ti, 3600)
    m, s = divmod(m, 60)
    s, ms = divmod(s, 1)
    return "{:2d}h{:2d}m{:2d}s.{:3d}".format(int(h), int(m), int(s), int(ms * 1000))


def register_environment(n_good, n_adv, n_landmarks, n_food, n_forests, init_weights, id_mapping=None):
    global N_GOOD, N_ADV, N_LAND, N_LANDMARKS, N_FOOD, N_FORESTS, ID_MAPPING, INIT_WEIGHTS
    N_GOOD = n_good
    N_ADV = n_adv
    N_LANDMARKS = n_landmarks
    N_FOOD = n_food
    N_FORESTS = n_forests
    N_LAND = N_LANDMARKS + N_FOOD + N_FORESTS
    INIT_WEIGHTS = init_weights
    ID_MAPPING = id_mapping
    # print("SHARE_WEIGHTS", SHARE_WEIGHTS)


def name_encode(name, convert):
    # print(name)

    def agent_decode(name):
        # if name == "self":
        #     return last
        match = re.match(r'agent_(\d+)', name)
        if match:
            return int(match.group(1))
        match = re.match(r'good(\d+)', name)
        if match:
            return int(match.group(1)) + N_ADV
        match = re.match(r'adv(\d+)', name)
        if match:
            return int(match.group(1))
        return name

    names = name.split('/')
    ret = []
    for name in names:
        decoded = agent_decode(name)
        # if type(decoded) == int:
        #     # decoded = id_reverse_mapping[decoded] if convert else decoded
        #     last = decoded
        ret.append(decoded)

    last = None

    is_new = None

    for i in range(len(ret)):
        if type(ret[i]) == int:
            if is_new is None:
                is_new = ret[i] in NEW_IDS
            if convert:
                ret[i] = ID_MAPPING[ret[i]]
            if last == ret[i]:
            # if last == ret[i]:
                return None, None
            else:
                last = ret[i]
            ret[i] = str(ret[i])

    # print('/'.join(ret))
    return '/'.join(ret), is_new


def add_perturbation(weight):
    if PERTURBATION is None:
        return weight
    std = np.std(weight)
    return weight + np.random.normal(0., PERTURBATION * std, weight.shape)


def make_env(scenario_name, arglist, benchmark=False):
    import importlib
    from mpe_local.multiagent.environment import MultiAgentEnv

    module_name = "mpe_local.multiagent.scenarios.{}".format(scenario_name)
    scenario_class = importlib.import_module(module_name).Scenario
    # load scenario from script
    # print(Scenario.__module__.__file__)
    ratio = 1.0 if arglist.map_size == "normal" else 2.0
    scenario = scenario_class(n_good=N_GOOD, n_adv=N_ADV, n_landmarks=N_LANDMARKS, n_food=N_FOOD, n_forests=N_FORESTS,
                              no_wheel=arglist.no_wheel, sight=arglist.sight, alpha=arglist.alpha, ratio=ratio)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data, export_episode=arglist.save_gif_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, done_callback=scenario.done, info_callback=scenario.info,
                            export_episode=arglist.save_gif_data)
    return env


def make_session(graph, num_cpu):
    # print("num_cpu:", num_cpu)
    tf_config = tf.ConfigProto(
        # device_count={"CPU": num_cpu},
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    return tf.Session(graph=graph, config=tf_config)
    # return tf.Session(target=server.target, graph=graph, config=tf_config)


def get_trainer(side, i, scope, env, obs_shape_n):
    trainer = MADDPGAgentMicroSharedTrainer
    policy = FLAGS.adv_policy if side == "adv" else FLAGS.good_policy
    share_weights = FLAGS.adv_share_weights if side == "adv" else FLAGS.good_share_weights
    if policy == "att-maddpg":
        model_p = partial(mlp_model_adv_p if side == "adv" else mlp_model_agent_p, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
        model_q = partial(mlp_model_adv_q if side == "adv" else mlp_model_agent_q, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
    elif policy == "PC":
        model_p = partial(mlp_model_adv_p_numbered if side == "adv" else mlp_model_agent_p_numbered, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
        model_q = partial(mlp_model_adv_q_numbered if side == "adv" else mlp_model_agent_q_numbered, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
    elif policy == "maddpg":
        model_p = mlp_model
        model_q = mlp_model
    elif policy == "mean_field":
        model_p = mlp_model
        model_q = partial(mean_field_adv_q_model if side == "adv" else mean_field_agent_q_model, n_good=N_GOOD,
                          n_adv=N_ADV, n_land=N_LAND, index=i)
    else:
        raise NotImplementedError
    # print(obs_shape_n)
    num_units = (FLAGS.adv_num_units if side == "adv" else FLAGS.good_num_units) or FLAGS.num_units
    return trainer(scope, model_p, model_q, obs_shape_n, env.action_space, i, FLAGS, num_units, local_q_func=False)


def get_adv_trainer(i, scope, env, obs_shape_n):
    return get_trainer("adv", i, scope, env, obs_shape_n)


def get_good_trainer(i, scope, env, obs_shape_n):
    return get_trainer("good", i, scope, env, obs_shape_n)


def show_size():
    s = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        tot = 1
        for dim in shape:
            tot *= dim
        # if tot > 5000:
        #     print(tot, shape, var.name)
        s += tot
    # print("total size:", s)


def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def load_weights(load_path):
    import joblib
    global CACHED_WEIGHTS

    CACHED_WEIGHTS.update(joblib.load(load_path))


def clean(d):
    rd = {}
    for k, v in d.items():
        # if v.shape[0] == 456 or v.shape[0] == 1552:
        #     print(k, v.shape)
        if type(k) == tuple:
            rd[k[0]] = v
        else:
            rd[k] = v
    return rd


def load_all_weights(load_dir, n):
    global CACHED_WEIGHTS
    CACHED_WEIGHTS = {}
    for i in range(n):
        # print(os.path.join(load_dir[i], "agent{}.trainable-weights".format(i)))
        load_weights(os.path.join(load_dir[i], "agent{}.trainable-weights".format(i)))
    # print(CACHED_WEIGHTS)
    CACHED_WEIGHTS = clean(CACHED_WEIGHTS)
    # for weight, value in CACHED_WEIGHTS.items():
    #     if weight[-7:] == "gamma:0" or weight[-6:] == "beta:0":
    #         print(weight, value)
    # print(CACHED_WEIGHTS.keys())


def parse_args(add_extra_flags=None):
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str,
                        default="food_collect", # food_collect, simple_spread
                        help="name of the scenario script")
    parser.add_argument("--map-size", type=str, default="normal")
    parser.add_argument("--sight", type=float, default=100)
    parser.add_argument("--no-wheel", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--show-attention", action="store_true", default=False)
    parser.add_argument("--max-episode-len", type=int,
                        default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int,
                        default=200000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int,
                        default=0, help="number of adversaries")
    parser.add_argument("--num-good", type=int,
                        default=3, help="number of good")
    parser.add_argument("--num-food", type=int,
                        default=3, help="number of food")
    parser.add_argument("--good-policy", type=str,
                        default="att-maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str,
                        default="att-maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float,
                        default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=32,
                        help="number of units in the mlp")
    parser.add_argument("--good-num-units", type=int)
    parser.add_argument("--adv-num-units", type=int)
    parser.add_argument("--n-cpu-per-agent", type=int, default=24)
    parser.add_argument("--good-share-weights", action="store_true", default=True)
    parser.add_argument("--adv-share-weights", action="store_true", default=True)
    parser.add_argument("--use-gpu", action="store_true", default=True)
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./result/simple_spread_epc",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--train-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--checkpoint-rate", type=int, default=0)
    parser.add_argument("--load-dir", type=str, default="./result/simple_spread_epc/stage-0/seed-0",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save-gif-data", action="store_true", default=False)
    parser.add_argument("--render-gif", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=10000,
                        help="number of iterations run for benchmarking")

    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--save-summary", action="store_true", default=False)
    parser.add_argument("--timeout", type=float, default=0.03)
    if add_extra_flags is not None:
        parser = add_extra_flags(parser)

    return parser.parse_args()


def calc_size(var_list):
    s = 0
    for var in var_list:
        shape = var.get_shape()
        tot = 1
        for dim in shape:
            tot *= dim
        # if tot > 5000:
        #     print(tot, shape, var.name)
        s += tot
    return s

# @ray.remote
class Agent(multiprocessing.Process):
    def __init__(self, index, n, obs_batch_size, obs_shape, update_event, save_event, save_dir, load_dir,
                 cached_weights, num_cpu, get_trainer, main_conn, obs_queue, env_conns, use_gpu, timeout,
                 attention_mode, agent_sends, agent_recvs, tmp_file, obs_len, act_len, batch_size, train):
        multiprocessing.Process.__init__(self, daemon=True)
        # self.sess = SESSIONS[i]
        # self.graph = GRAPHS[i]
        self.index = index
        self.n = n
        self.obs_batch_size = obs_batch_size
        self.obs_shape = obs_shape
        self.scope = "agent_runner_{}".format(index)
        self.num_cpu = num_cpu
        self.get_trainer = get_trainer
        self.main_conn = main_conn
        self.obs_queue = obs_queue
        self.env_conns = env_conns
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.trainer = None
        self.sess = None
        self.graph = None
        self.var_list = None
        self.cached_weights = cached_weights
        self.use_gpu = use_gpu
        self.update_event = update_event
        self.save_event = save_event
        self.sum_batch_size = None
        self.tot_batch = None
        self.timeout = timeout
        self.attention_mode = attention_mode
        self.agent_sends = agent_sends
        self.agent_recvs = agent_recvs
        self.tmp_file = open(tmp_file, "rb")
        self.obs_len = obs_len
        self.act_len = act_len
        self.batch_size = batch_size
        self.train = train
        # self.replay_buffer = ReplayBuffer(int(1e6))

    # def get_trainer(self):
    #     return self.trainer

    def build(self):
        # U.set_session(self.sess)

        # with self.graph.as_default():
        #     with self.sess:
        self.trainer = self.get_trainer()
        # print(self.trainer())
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(self.index, len(self.trainable_var_list), calc_size(self.trainable_var_list))
        # for var in self.trainable_var_list:
        #     print(self.graph, var.graph, var.name)

        return None

    def restore_weights(self, load_dir):
        # print(self.index, len(self.var))
        U.load_variables(os.path.join(load_dir, "agent{}.weights".format(self.index)),
                         variables=self.var_list,
                         sess=self.sess)
        return None

    def restore_trainable_weights(self, weights):
        restores = []
        for v in self.trainable_var_list:
            name, is_new = name_encode(v.name, convert=True)
            if name is None:
                continue
            w = weights[name]
            if is_new:
                w = add_perturbation(w)
            restores.append(v.assign(w))
        self.sess.run(restores)

    def action(self, obs):
        return self.trainer.batch_action(obs)

    def get_attn(self, obs):
        return self.trainer.batch_attn(obs)

    def target_action(self, batch_obs):
        # print(batch_obs)
        return self.trainer.target_action(batch_obs)


    def _run(self):
        self.sum_batch_size = 0
        self.tot_batch = 0
        warned = False
        self.graph = graph = tf.Graph()
        # n_envs = len(self.env_conns)
        with graph.as_default():
            self.sess = sess = make_session(graph, self.num_cpu)
            with sess:
                self.build()
                sess.run(tf.variables_initializer(self.var_list))

                if self.load_dir is not None:
                    self.restore_weights(load_dir=self.load_dir)
                elif self.cached_weights is not None:
                    self.restore_trainable_weights(self.cached_weights)
                    del self.cached_weights

                if self.train:
                    self.save_weights([os.path.join(self.save_dir, "episode-0")])

                self.main_conn.send(None)
                self.main_conn.recv()
                attention_mode = self.attention_mode

                while True:
                    obs_batch = np.zeros(shape=(self.obs_batch_size, *self.obs_shape))
                    receiver = [None for _ in range(self.obs_batch_size)]

                    cnt = 0
                    while cnt < self.obs_batch_size:
                        try:
                            obs_batch[cnt], receiver[cnt] = self.obs_queue.get(block=True, timeout=self.timeout)
                            # print(receiver[cnt])
                            cnt += 1
                        except queue.Empty:
                            break

                    self.sum_batch_size += cnt
                    self.tot_batch += 1
                    if cnt > 0:
                        # print(cnt, receiver[:cnt])
                        action = self.action(obs_batch[:cnt])
                        if attention_mode:
                            good_attn, adv_attn = self.get_attn(obs_batch[:cnt])
                        for i in range(cnt):
                            if attention_mode:
                                self.env_conns[receiver[i]].send((action[i], good_attn[i], adv_attn[i]))
                            else:
                                self.env_conns[receiver[i]].send(action[i])

                    if self.save_event.is_set():
                        episode = self.main_conn.recv()
                        save_dirs = [self.save_dir]
                        if episode is not None:
                            save_dirs.append(os.path.join(self.save_dir, "episode-{}".format(episode)))
                        self.main_conn.send(self.save_weights(save_dirs))
                        self.main_conn.recv()

                    if self.update_event.is_set():
                        # if self.n == 40:
                        #     print("#{} start update".format(self.index))
                        if self.sum_batch_size / self.tot_batch / self.obs_batch_size < .5 and not warned:
                            warned = True
                            print("Batch load insufficient ({:.2%})! Consider higher timeout!".format(self.sum_batch_size / self.tot_batch / self.obs_batch_size))
                        sampled_index, data_length = self.main_conn.recv()

                        total_numbers = sum(self.obs_len) + sum(self.act_len) + self.n + sum(self.obs_len) + self.n
                        float_length = 4
                        assert total_numbers * float_length == data_length

                        obs_n = [np.zeros((self.batch_size, self.obs_len[i])) for i in range(self.n)]
                        action_n = [np.zeros((self.batch_size, self.act_len[i])) for i in range(self.n)]
                        reward = np.zeros(self.batch_size)
                        obs_next_n = [np.zeros((self.batch_size, self.obs_len[i])) for i in range(self.n)]
                        done = np.zeros(self.batch_size)

                        # print(sampled_index)
                        for i, index in enumerate(sampled_index):
                            self.tmp_file.seek(index * data_length)
                            flat_data = bytes_to_exp(self.tmp_file, total_numbers)

                            last = 0
                            for j in range(self.n):
                                l = self.obs_len[j]
                                obs_n[j][i], last = flat_data[last: last + l], last + l

                            for j in range(self.n):
                                l = self.act_len[j]
                                action_n[j][i], last = flat_data[last: last + l], last + l

                            reward[i] = flat_data[last + self.index]
                            last += self.n

                            for j in range(self.n):
                                l = self.obs_len[j]
                                obs_next_n[j][i], last = flat_data[last: last + l], last + l

                            done[i] = flat_data[last + self.index]
                            assert last + self.n == total_numbers

                        target_obs = []
                        for i in range(self.n):
                            if i < self.index:
                                target_obs.append(self.agent_recvs[i].recv())
                                self.agent_sends[i].send(obs_next_n[i])
                            elif i == self.index:
                                target_obs.append(obs_next_n[i])
                            else:
                                self.agent_sends[i].send(obs_next_n[i])
                                target_obs.append(self.agent_recvs[i].recv())

                        target_actions = self.target_action(np.concatenate(target_obs, axis=0))

                        # print(target_actions.shape)

                        target_action_n = [None for _ in range(self.n)]
                        for i in range(self.n):
                            ta = target_actions[i * self.batch_size: (i + 1) * self.batch_size]
                            if i < self.index:
                                target_action_n[i] = self.agent_recvs[i].recv()
                                self.agent_sends[i].send(ta)
                            elif i == self.index:
                                target_action_n[i] = ta
                            else:
                                self.agent_sends[i].send(ta)
                                target_action_n[i] = self.agent_recvs[i].recv()

                        if self.n == 40:
                            # print("#{} target action done".format(self.index))
                            self.main_conn.recv()

                        self.update(((obs_n, action_n, reward, obs_next_n, done), target_action_n))

                        if self.n == 40:
                            gc.collect()


                        self.main_conn.send(None)
                        self.main_conn.recv()

    def run(self):
        # self.server = tf.train.Server(CLUSTER, job_name="local%d" % self.index, task_index=0)

        if self.use_gpu:
            # print("GPU")
            with tf.device("/device:GPU:%d" % self.index):
                self._run()
        else:
            with tf.device("/cpu:0"):
                self._run()

    def save_weights(self, save_dirs):
        import joblib

        all_vars = self.sess.run(self.var_list)
        all_save_dict = {v.name: value for v, value in zip(self.var_list, all_vars)}
        trainable_save_dict = {name_encode(v.name, convert=False)[0]: all_save_dict[v.name]
                               for v in self.trainable_var_list}
        for save_dir in save_dirs:
            all_save_path = os.path.join(save_dir, "agent{}.weights".format(self.index))
            touch_path(all_save_path)
            # print(len(save_dict))
            joblib.dump(all_save_dict, all_save_path)

            trainable_save_path = os.path.join(save_dir, "agent{}.trainable-weights".format(self.index))
            touch_path(trainable_save_path)
            joblib.dump(trainable_save_dict, trainable_save_path)

        return trainable_save_dict

    def preupdate(self):
        self.trainer.preupdate()
        return None

    def update(self, args):
        data, target_act_next_n = args
        # _obs_n, _action_n, _rewards, _obs_next_n, _dones = data
        self.trainer.update(data, target_act_next_n)
        # return [0.] * 5


class Environment(multiprocessing.Process):
    def __init__(self, env, index, max_len, actor_queues, actor_conns, main_conn, experience_queue, attention_mode):
        multiprocessing.Process.__init__(self, daemon=True)

        self.env = env
        self.index = index
        self.n = env.n
        self.max_len = max_len
        self.actor_queues = actor_queues
        self.actor_conns = actor_conns
        self.main_conn = main_conn
        self.experience_queue = experience_queue
        self.attention_mode = attention_mode

    def run(self):
        env, n, actor_queues, actor_conns, experience_queue = \
            self.env, self.n, self.actor_queues, self.actor_conns, self.experience_queue

        attention_mode = self.attention_mode
        while True:
            obs_n = env.reset()
            steps = 0
            sum_reward_n = [0.] * n
            while True:
                for i in range(n):
                    actor_queues[i].put((obs_n[i], self.index))

                action_n = []
                good_attn_n = []
                adv_attn_n = []
                for i in range(n):
                    recv = actor_conns[i].recv()
                    if attention_mode:
                        action, good_attn, adv_attn = recv
                        good_attn_n.append(good_attn)
                        adv_attn_n.append(adv_attn)
                    else:
                        action = recv
                    action_n.append(action)

                new_obs_n, reward_n, done_n, info_n = env.step(action_n)
                for i in range(n):
                    sum_reward_n[i] += reward_n[i]
                    reward_n[i] = np.array(reward_n[i])
                    done_n[i] = np.array(done_n[i])

                steps += 1
                end_of_episode = steps > self.max_len or all(done_n)
                experience_queue.put([self.index, obs_n, action_n, reward_n, new_obs_n, done_n, end_of_episode, sum_reward_n, info_n, good_attn_n, adv_attn_n])

                if end_of_episode:
                    num_episodes = self.main_conn.recv()
                    # print("num_episodes:", num_episodes)
                    if num_episodes is not None:
                        # print("Saving gif!")
                        memory = env.export_memory()
                        # print(self.index, len(memory), num_episodes)
                        joblib.dump(memory, os.path.join(FLAGS.save_dir, "episode-{}.gif-data".format(num_episodes)))
                        # print("gif saved.", self.index)
                        self.main_conn.send(None)

                if True:
                    pause = self.main_conn.recv()
                    if pause:
                        self.main_conn.recv()
                    if end_of_episode:
                        break
                    else:
                        obs_n = new_obs_n



def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    trainer = MADDPGAgentMicroSharedTrainer
    for i in range(num_adversaries):
        trainer = get_adv_trainer(i, "adv{}".format(i), env, obs_shape_n)
        trainers.append(trainer)
    for i in range(num_adversaries, env.n):
        trainer = get_good_trainer(i, "good{}".format(i - num_adversaries), env, obs_shape_n)
        trainers.append(trainer)
    return trainers


def eval(arglist):
    with U.single_threaded_session():
        global FLAGS
        FLAGS = arglist
        save_gifs = 1

        # Create environment
        curriculum = 0

        id_mapping = None
        register_environment(n_good=FLAGS.num_good, n_adv=FLAGS.num_adversaries, n_landmarks=0, n_food=FLAGS.num_food,
                             n_forests=0, init_weights=curriculum, id_mapping=id_mapping)
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers

        if save_gifs:
            frames = []
            # frames.append(env.render('rgb_array')[0])

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))


        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            print(arglist.load_dir)
            # U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            if (train_step % arglist.max_episode_len == 0):
                history_info = info_n
            else:
                iitem = 0
                for item in history_info:
                    history_info[item] += info_n[item] # 바꿈
                    iitem += 1

            if not train_step:
                info_all = history_info



            episode_step += 1
            done = all(done_n)

            # terminal = (episode_step >= arglist.max_episode_len)

            num = len(episode_rewards)//10000
            savedir = "./simple_sheep_wolf/5_3_2_4_50_curr"+str(num)+"/"
            max_len = 50+num*25

            if not curriculum:
                terminal = (episode_step >= arglist.max_episode_len)
            else:
                terminal = (episode_step >= max_len)


            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                iitem = 0
                for item in history_info:
                    info_all[item] += history_info[item]
                    iitem += 1
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies

            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            # print(done_n)
            # print(info_n)
            '''
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                                                                                          train_step, len(episode_rewards), np.mean(episode_rewards[-2:]), round(time.time()-t_start, 3)))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                                                                                                                    train_step, len(episode_rewards), np.mean(episode_rewards[-2:]),
                                                                                                                    [np.mean(rew[-2:]) for rew in agent_rewards], round(time.time()-t_start, 3)))

            '''

            if arglist.display:
                if save_gifs:
                    frames.append(env.render('rgb_array')[0])
                time.sleep(0.1)
                env.render()
                if terminal or done:
                    print (history_info)
                    if num_adversaries == 0:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-2:]), round(time.time()-t_start, 3)))
                    else:
                        print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-2:]),
                            [np.mean(rew[-2:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                if len(episode_rewards) > arglist.num_episodes:
                    break

                continue
            '''
            if terminal and (len(episode_rewards) % arglist.save_rate == 0) and len(episode_rewards)>1000:
                time.sleep(0.1)
                env.render()
                continue
            '''

            # update all trainers, if not in display or benchmark mode
            # loss = None
            for agent in trainers:
                agent.preupdate()
            # for agent in trainers:
            #     loss = agent.update(trainers, train_step) #all agent update


            # save model, display training output

            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                if not curriculum:
                    U.save_state(arglist.save_dir, saver=saver)
                else:
                    U.save_state(savedir, saver=saver)
                print (info_all)
                info_all = info_n
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later

            if len(episode_rewards) > arglist.num_episodes or train_step>3000000:
                print('end')

                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break
        if save_gifs:
            gif_num = 0
            imageio.mimsave(arglist.load_dir+'5_3_2_4normal_noshape.gif',
                            frames, duration=1/5)

if __name__ == "__main__":
    arglist = parse_args()
    eval(arglist)