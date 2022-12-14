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
                            scenario.observation, scenario.benchmark_data, export_episode=arglist.save_gif_data, noise_std=arglist.noise_std)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, done_callback=scenario.done, info_callback=scenario.info,
                            export_episode=arglist.save_gif_data, noise_std=arglist.noise_std)
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
    elif policy == "r-maddpg":
        model_nature = mlp_model
        model_p = mlp_model
        model_q = mlp_model
    elif policy == "r-att-maddpg":
        model_nature = partial(mlp_model_adv_p if side == "adv" else mlp_model_agent_p, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
        model_p = partial(mlp_model_adv_p if side == "adv" else mlp_model_agent_p, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
        model_q = partial(mlp_model_adv_q if side == "adv" else mlp_model_agent_q, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
    elif policy == "mean_field":
        model_p = mlp_model
        model_q = partial(mean_field_adv_q_model if side == "adv" else mean_field_agent_q_model, n_good=N_GOOD,
                          n_adv=N_ADV, n_land=N_LAND, index=i)
    else:
        raise NotImplementedError

    num_units = (FLAGS.adv_num_units if side == "adv" else FLAGS.good_num_units) or FLAGS.num_units
    if policy == "r-maddpg" or policy == "r-att-maddpg":
        trainer = RMADDPGAgentMicroSharedTrainer
        return trainer(scope, model_nature, model_p, model_q, obs_shape_n, env.action_space, i, FLAGS, num_units, local_q_func=False)
    else:
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
                        default="r-att-maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str,
                        default="r-att-maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--lr-nature", type=float, default=1e-3,
                        help="learning rate for Adam optimizer of nature actor")
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
    parser.add_argument("--noise-std", type=float, default=0.0)
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./result/simple_spread_epc",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--train-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--checkpoint-rate", type=int, default=0)
    parser.add_argument("--load-dir", type=str, default="./result/epc_rmaddpg_noise0/stage-0/seed-0",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save-gif-data", action="store_true", default=False)
    parser.add_argument("--render-gif", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--benchmark-iters", type=int, default=10000,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./result/validation")
    parser.add_argument("--exp-name", type=str, default="epc")
    parser.add_argument("--save-gifs", action="store_true", default=False)

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


def restore_weights(i, load_dir):
    # print(self.index, len(self.var))
    U.load_variables(os.path.join(load_dir, "agent{}.weights".format(i)),
                     variables=None, sess=None)
    return None

def eval(arglist):
    with U.single_threaded_session():
        global FLAGS
        FLAGS = arglist
        # save_gifs = 1

        # Create environment
        curriculum = 0

        id_mapping = None
        register_environment(n_good=FLAGS.num_good, n_adv=FLAGS.num_adversaries, n_landmarks=0, n_food=FLAGS.num_food,
                             n_forests=0, init_weights=curriculum, id_mapping=id_mapping)
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers

        if arglist.save_gifs:
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

            for i, agent in enumerate(trainers):
                print(agent)
                restore_weights(i, load_dir=arglist.load_dir)
                # U.load_state(os.path.join(arglist.load_dir, "agent{}.weights".format(i)))
            # self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # U.load_variables(os.path.join(arglist.load_dir, "agent{}.weights".format(self.index)), variables=self.var_list, sess=self.sess)

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
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)] # agent action
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
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal) # agent experience
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                print('')
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

                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                      train_step, len(episode_rewards), episode_rewards[-2]/arglist.num_good, round(time.time()-t_start, 3)))

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            # print(done_n)
            # print(info_n)
            # if num_adversaries == 0:
            #     print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
            #           train_step, len(episode_rewards), np.mean(episode_rewards[-2:])/arglist.num_good, round(time.time()-t_start, 3)))

            # for benchmarking learned policies
            if arglist.benchmark:
                if len(episode_rewards) > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + "_"+ arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')

                    assert arglist.num_good != 0
                    episode_rewards = episode_rewards[:-1]
                    episode_rewards_mean =  [i/arglist.num_good for i in episode_rewards]
                    print(np.mean(episode_rewards_mean))
                    with open(file_name, 'wb') as fp:
                        pickle.dump(episode_rewards_mean, fp)
                    break
                continue



            if arglist.display:
                if arglist.save_gifs:
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
        if arglist.save_gifs:
            gif_num = 0
            imageio.mimsave(arglist.load_dir+'5_3_2_4normal_noshape.gif',
                            frames, duration=1/5)

if __name__ == "__main__":
    arglist = parse_args()
    eval(arglist)