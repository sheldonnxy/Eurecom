from runner import Runner
from Libs.Environments.DataCollection import DataCollection
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from common.Defines import city, radio_ch_model, device_position, devices_params, \
    agent_params, uav_start_pose, uav_terminal_pose, ch_param
from Libs.Environments.DataCollection import DataCollection
from ChannelEstimator import *
import json
import random, numpy as np, torch


if __name__ == '__main__':
    for i in range(1):
    # for i in range(8):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)

        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        learning_channel_model = None
        if args.model:
            ch_hidden_layers = [(50, 'tanh'), (20, 'linear')]
            learning_channel_model = SLAL(buffer_size=10000, channel_param=ch_param, hidden_layers=ch_hidden_layers,
                                          city=city)

        env = DataCollection(args,
                             learning_channel_model=learning_channel_model,
                             city=city,
                             radio_ch_model=radio_ch_model,
                             device_position=device_position,
                             devices_params=devices_params,
                             agent_params=agent_params,
                             uav_start_pose=uav_start_pose,
                             uav_terminal_pose=uav_terminal_pose)

        # env = StarCraft2Env(map_name=args.map,
        #                     step_mul=args.step_mul,
        #                     difficulty=args.difficulty,
        #                     game_version=args.game_version,
        #                     replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        runner = Runner(env, args)

        if not args.evaluate:
            runner.run(i, model=args.model)
            log_path = args.result_dir + '/' + args.alg + '/' + args.map + '/' + 'log_params.txt'
            with open(log_path, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        else:
            collected_data, _ = runner.evaluate()
            print('The collected data of {} is  {}'.format(args.alg, collected_data))
            break
        # env.close()
