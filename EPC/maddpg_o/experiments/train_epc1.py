import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from .train_helper.train_helpers import parse_args
from .train_helper.proxy_train import proxy_train
from .train_mix_match import mix_match
from .compete import compete
import os
import joblib
import numpy as np
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def add_extra_flags(parser):
    parser.add_argument('--cooperative', action="store_true", default=True)
    parser.add_argument('--initial-population', type=int, default=3)#6
    parser.add_argument('--num-selection', type=int, default=2)#3
    parser.add_argument('--num-stages', type=int, default=3)
    parser.add_argument('--stage-num-episodes', type=int, nargs="+", default=[25, 25, 25])
    parser.add_argument('--stage-n-envs', nargs="+", default=[25]) 
    parser.add_argument('--test-num-episodes', type=int, default=2000)
    # parser.add_argument('--test-standard', type=str, default="average")
    parser.add_argument('--mutation-rate', type=float, default=0.2)
    parser.add_argument('--roulette-mode', type=str, default='proportional')
    return parser


def touch_dir(dirname):
    os.makedirs(dirname, exist_ok=True)


def join_dir(dirname1, dirname2):
    dirname = os.path.join(dirname1, dirname2)
    touch_dir(dirname)
    return dirname


def expand(arr, n):
    return arr + [arr[-1]] * (n - len(arr))


def train_epc(arglist):
    import copy
    original_arglist = copy.deepcopy(arglist)

    cooperative = original_arglist.cooperative
    num_stages = original_arglist.num_stages
    stage_num_episodes = expand(original_arglist.stage_num_episodes, num_stages)
    stage_n_envs = expand(original_arglist.stage_n_envs, num_stages)

    save_dir = arglist.save_dir

    stage_dir = join_dir(save_dir, "stage-0")
    arglist.num_episodes = stage_num_episodes[0]
    arglist.n_envs = stage_n_envs[0]

    n = original_arglist.initial_population
    roulette_mode = original_arglist.roulette_mode

    last_dirs = []
    print("Training stage-0 ...")
    for i in range(n):
        arglist.save_dir = join_dir(stage_dir, "seed-{}".format(i))
        last_dirs.append(arglist.save_dir)
        print("Training seed-{}".format(i))
        proxy_train({"arglist": arglist})

    k = original_arglist.num_selection

    def compete_last():
        compete_arglist = copy.deepcopy(arglist)
        compete_arglist.parallel_limit = 1
        compete_arglist.symmetric = True
        compete_arglist.antisymmetric = False
        compete_arglist.dot_product = False
        compete_arglist.double = False
        compete_arglist.competitor_load_dirs = [last_dirs[0]] if cooperative else last_dirs
        compete_arglist.baseline_load_dirs = last_dirs
        compete_arglist.competitor_share_weights = [True]
        compete_arglist.baseline_share_weights = [True]
        compete_arglist.competitor_models = ["att-maddpg"]
        compete_arglist.baseline_models = ["att-maddpg"]
        compete_arglist.competitor_num_units = [original_arglist.num_units]
        compete_arglist.baseline_num_units = [original_arglist.num_units]
        compete_arglist.competitor_checkpoint_rates = None
        compete_arglist.baseline_checkpoint_rates = None
        compete_arglist.save_dir = join_dir(stage_dir, "compete_result")
        return compete(compete_arglist)

    def seed_roulette(scores, mode='proportional'):
        '''
        mode = 'proportional' or 'ranking'
        'proportional'
            proportional wheel after normalize
            zero probability for lowest
        'ranking'
            fitness is reciprocal of rank
        '''
        if mode == 'proportional':
            min_fitness = min([s for s in scores])
            fitness = [s+np.abs(min_fitness) for s in scores]
            total_fitness = sum(fitness)
        elif mode == 'ranking':
            s = sorted(scores, reverse=True)
            rank = [s.index(x)+1 for x in scores]
            fitness = [1/r for r in rank]
            total_fitness = sum(fitness)

        par = []  
        for i in range(2):
            random_rate = random.uniform(0, 1)  # roulette 
            k = 0
            sum_fitness = 0
            while k < len(fitness):
                sum_fitness += fitness[k]
                if random_rate <= sum_fitness/total_fitness:
                    break
                k += 1
            par.append(k)
        return par

    for s in range(num_stages - 1):
        print("Competing stage-{} ...".format(s))
        report = compete_last()

        if not cooperative:
            wolf_scores = np.average(report["competitor_wolf_scores"], axis=-1)
            wolf_indices = wolf_scores.argsort()[-k:][::-1].tolist()
            sheep_scores = np.average(report["baseline_sheep_scores"], axis=-1)
            sheep_indices = sheep_scores.argsort()[-k:][::-1].tolist()

            print("Top wolf seeds:", wolf_indices)
            print("Top sheep seeds:", sheep_indices)
        else:
            scores = np.average(report["baseline_sheep_scores"], axis=-1)
            indices = scores.argsort()[-k:][::-1].tolist() # sorted list by fitness
        stage_dir = join_dir(save_dir, "stage-{}".format(s + 1))
        arglist.num_episodes = stage_num_episodes[s + 1]
        arglist.n_envs = stage_n_envs[s + 1]

        cur_dirs = []
        n = 0
        print("Training stage-{} ...".format(s + 1))

        # evolutionary learning
        if not cooperative:
            for i1 in range(k):
                for j1 in range(i1, k):
                    for i2 in range(k):
                        for j2 in range(i2, k):
                            print("Training seed-{} (from wolf seed {} x {} and sheep seed {} x {}) ...".format(n, wolf_indices[i1], wolf_indices[j1], sheep_indices[i2], sheep_indices[j2]))
                            arglist.save_dir = join_dir(stage_dir, "seed-{}".format(n))
                            cur_dirs.append(arglist.save_dir)
                            arglist.wolf_init_load_dirs = [last_dirs[wolf_indices[i1]], last_dirs[wolf_indices[j1]]]
                            arglist.sheep_init_load_dirs = [last_dirs[sheep_indices[i2]], last_dirs[sheep_indices[j2]]]
                            mix_match(copy.deepcopy(arglist))
                            n += 1
            assert (n == (k * (k + 1) // 2) ** 2)
        else:
            agent_scores = [report['detailed_reports'][i]['sheep']['ind_score'] for i in range(num_stages)]
            arglist.num_food *= 2
            ## roulette
            parents_set = []
            while n < (k * (k + 1) // 2):
                par1, par2 = seed_roulette(scores, mode=roulette_mode)
                parents_set.append(tuple([par1, par2]))
                # if tuple([par1, par2]) in parents_set:
                #     continue
                print("Training seed-{} (from seed {} x {}) ...".format(n, par1, par2))
                arglist.save_dir = join_dir(stage_dir, "seed-{}".format(n))
                cur_dirs.append(arglist.save_dir)
                arglist.wolf_init_load_dirs = [last_dirs[par1], last_dirs[par2]]
                arglist.sheep_init_load_dirs = [last_dirs[par1], last_dirs[par2]]
                arglist.agent_scores = [agent_scores[par1], agent_scores[par2]]
                mix_match(copy.deepcopy(arglist))
                n += 1
            assert (n == (k * (k + 1) // 2))
        
        last_dirs = cur_dirs

        arglist.num_good *= 2
        if not cooperative:
            arglist.num_adversaries *= 2

    print("Competing final stage ...")
    compete_last()
    print("Done. Final result stored in {}/report.json.".format(join_dir(stage_dir, "compete_result")))

if __name__ == "__main__":
    arglist = parse_args(add_extra_flags)

    train_epc(arglist)
