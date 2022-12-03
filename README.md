# GT Project

## Installation

- conda environment setting
```
conda create -n GT python=3.7 -y
conda activate GT
```
- nessary packages
```
pip install tensorflow==1.13.1
pip install protobuf==3.20
pip install numpy==1.16.5
pip install gym==0.13.0
pip install joblib imageio
```

## Quick start

```
python EPC/maddpg_o/experiments/train_epc1.py 2>&1 | tee epc_simplespread.txt
python EPC/maddpg_o/experiments/train_epc1.py --good-policy r-att-maddpg
```


## Case study: Multi-Agent Particle Environments

We demonstrate here how the code can be used in conjunction with the(https://github.com/qian18long/epciclr2020/tree/master/mpe_local). It is based on(https://github.com/openai/multiagent-particle-envs)

## Paper citation

```
@inproceedings{epciclr2020,
  author = {Qian Long and Zihan Zhou and Abhinav Gupta and Fei Fang and Yi Wu and Xiaolong Wang},
  title = {Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning},
  booktitle = {International Conference on Learning Representations},
  year = {2020}
}
```
