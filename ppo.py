import argparse
import os
import distutils.util as strtobool
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from stable_baselines3.common.env_util import make_vec_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--exp-name', type = str, 
            default = os.path.basename(__file__).rstrip('.py'), 
            help = 'Name of experiment'
    )
    parser.add_argument(
            '--gym-id', type = str, 
            default = 'LunarLander-v2', 
            help = 'String to instantiate the gym environment'
    )
    parser.add_argument(
            '--learning-rate', type = float, 
            default = 3e-4,
            help = 'Learning rate for optimizer'
    )
    parser.add_argument(
            '--seed', type = int, 
            default = 1337,
            help = 'Initialize seed for reproducibility'
    )
    parser.add_argument(
            '--max-timesteps', type = int, 
            default = 100_000,
            help = 'Maximum timesteps to run training'
    )
    parser.add_argument(
            '--torch-deterministic', type = lambda x: bool(strtobool(x)),
            default = True,
            nargs = '?', 
            const = True,
            help = 'Sets torch.backends.cudnn.deterministic to bool value'
    )
    parser.add_argument(
            '--cuda-enabled', type = lambda x: bool(strtobool(x)), 
            default = True,
            nargs = '?', 
            const = True,
            help = 'if True enables use of CUDA'
    )

    parser.add_argument(
            '--num-envs', type = int, 
            default = 4,
            help = 'Number of vectorized environments'
    )
    parser.add_argument(
            '--num-steps', type = int, 
            default = 256,
            help = 'Number of steps to run each env for each rollout'
    )
    parser.add_argument(
            '--anneal-lr', type = lambda x: bool(strtobool(x)), 
            default = True,
            nargs = '?', 
            const = True,
            help = 'Toggle for lr annealing for both actor and critic networks'
    )
    parser.add_argument(
            '--gae', type = lambda x: bool(strtobool(x)), 
            default = True,
            nargs = '?', 
            const = True,
            help = 'Toggle for implementation of General Advantage estimation'
    )
    parser.add_argument(
            '--gae-lambda', type = float, 
            default = 0.95,
            help = 'Lambda exponentially weighted average of Q approximations E[r0 + gamma*V(s+1)](1-lambda)lambda'
    )
    parser.add_argument(
            '--gamma', type = float, 
            default = 0.99,
            help = 'Discount factor used for subsequent time steps'
    )
    parser.add_argument(
            '--num-minibatches', type = int, 
            default = 4,
            help = 'Number of minibatches used for training'
    )
    parser.add_argument(
            '--update-epochs', type = int, 
            default = 4,
            help = 'Number of epochs to train policy with'
    )
    parser.add_argument(
            '--norm-adv', type = lambda x: bool(strtobool(x)), 
            default = True,
            nargs = '?', 
            const = True,
            help = 'Toggle for Advantage normalization'
    )
    parser.add_argument(
            '--clip-coef', type = float, 
            default = 0.2,
            help = 'Clipping coefficient epsilon used for surrogate loss function'
    )
    parser.add_argument(
            '--clip-vloss', type = lambda x: bool(strtobool(x)), 
            default = True,
            nargs = '?', 
            const = True,
            help = 'Toggle for clipped value loss function as implemented in paper'
    )
    parser.add_argument(
            '--ent-coef', type = float, 
            default = 0.01,
            help = 'Coefficient for entropy'
    )
    parser.add_argument(
            '--vf-coef', type = float, 
            default = 0.5,
            help = 'Coefficient of the value function'
    )
    parser.add_argument(
            '--max-grad-norm', type = float, 
            default = 0.5,
            help = 'Maximum norm for the gradient clipping'
    )
    parser.add_argument(
            '--target-kl', type = float, 
            default = None,
            help = 'Using KL divergence for early stopping as implemented in spinning up'
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs*args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == '__main__':
    args = parse_args()
    run_name = f'{args.gym_id}__{args.exp_name}__{datetime.now().strftime("%d-%m-%Y__%H-%M-%S")}'
    writer = SummaryWriter(f'logs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '  \n'.join([f'{key}: {value}' for key, value in vars(args).items()])
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda_enabled else 'cpu')
    
    def layer_init(layer, std=np.sqrt(2), bias=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)
        return layer

    class Agent(nn.Module):
        def __init__(self, envs):
            super(Agent, self).__init__()

            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
            ) # Actor learns the policy for the critic to evaluate
            
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            ) # Critic acts as a value function to criticize the policies actions
        
        def get_action_value(self, x, action=None):
            logits = self.actor(x)
            probs = torch.distributions.categorical.Categorical(logits = logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        
        def get_value(self, x):
            return self.critic(x)



    #envs = gym.vector.SyncVectorEnv([make_vec_env(args.gym_id, args.seed+i, i, run_name) for i in range(args.num_envs)])
    envs = make_vec_env(args.gym_id, n_envs=args.num_envs, seed=args.seed)
    assert isinstance(envs.action_space, gym.spaces.Discrete), 'Only discrete actions space is supported for this PPO implementation.'
    print('{:=^40}'.format(f'{args.gym_id}'))
    print(f'Shape of Observation space: {envs.observation_space.shape}')
    print(f'Number of actions: {envs.action_space.n} \n\n')

    agent = Agent(envs).to(device)
    print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # initialize rollout storage
    observation_memory = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    action_memory = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    reward_memory = torch.zeros((args.num_steps, args.num_envs)).to(device)
    done_memory = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_memory = torch.zeros((args.num_steps, args.num_envs)).to(device)
    log_probs_memory = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    next_observation = torch.tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.max_timesteps // args.batch_size

    for update in tqdm(range(1, num_updates + 1), "Total Policy Updates"):
        if args.anneal_lr:
            frac = 1.0 - ((update - 1.0) / num_updates)
            new_lr = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = new_lr
        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            observation_memory[step] = next_observation
            done_memory[step] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_value(next_observation)
                value_memory[step] = value.flatten()
            action_memory[step] = action
            log_probs_memory[step] = log_prob

            next_observation, reward, done, info = envs.step(action.cpu().numpy())
            reward_memory[step] = torch.tensor(reward).to(device).view(-1)
            next_observation = torch.tensor(next_observation).to(device)
            next_done = torch.Tensor(done).to(device)

            for item in info:
                if 'episode' in item.keys():
                    #print(f'Global step: {global_step}  |  Episode Returns: {item["episode"]["r"]}')
                    writer.add_scalar("charts/episode_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episode_length", item["episode"]["l"], global_step)
                    break
        
        # Bootstrap if not done
        with torch.no_grad():
            next_values = agent.get_value(next_observation).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(reward_memory).to(device)
                last_gae_lambda = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        nextvalues = next_values
                    else:
                        next_non_terminal = 1.0 - done_memory[t+1]
                        nextvalues = value_memory[t+1]
                    delta = reward_memory[t] + args.gamma * nextvalues * next_non_terminal - value_memory[t]
                    advantages[t] = last_gae_lambda = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lambda
                returns = advantages + value_memory
            else:
                returns = torch.zeros_like(reward_memory).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_return = next_values
                    else:
                        next_non_terminal = 1.0 - done_memory[t+1]
                        next_return = returns[t+1]
                    returns[t] = reward_memory[t] + args.gamma * next_non_terminal * next_return
                advantages = returns - value_memory
        
        b_observations = observation_memory.reshape((-1,) + envs.observation_space.shape)
        b_log_probs = log_probs_memory.reshape(-1)
        b_actions = action_memory.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_memory.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_values = agent.get_action_value(b_observations[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_log_probs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # approximation of KL divergence
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss clipping
                new_values = new_values.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + torch.clamp(new_values - b_values[mb_inds], -args.clip_coef, args.clip_coef)

                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # if found: print("Hello World!")
        # if True: return True
        # tensorboard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        #writer.add_scalar("losses/clipfracs", clipfracs.mean(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    envs.close()
    writer.close()
