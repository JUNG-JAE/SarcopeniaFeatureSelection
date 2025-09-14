import argparse
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from ddqn_agent import Transition, DQNAgent, LinearEpsilon
from environment import FeatureSelectionEnv, FeatureSelectionEnvSarcopenia, impartial_rewards_accuracy
from utils_system import set_seed, save_plots, create_run_dir, set_logger, print_log


def main(args, device):
    set_seed(args.seed)
    
    # Logging setup
    run_dir = create_run_dir("./runs")
    (run_dir / "figs").mkdir(exist_ok=True, parents=True)
    (run_dir / "csv").mkdir(exist_ok=True, parents=True)
    logger = set_logger(run_dir)
    
    if args.dataset == "breast_cancer":
        env = FeatureSelectionEnv(test_size=args.test_size, cat_iterations=args.cat_iterations, cat_random_state=args.seed, k_fold=args.k_fold, random_state=args.seed, run_dir=run_dir)
    elif args.dataset == "sarcopenia":
        env = FeatureSelectionEnvSarcopenia(data_path=args.sarc_data_path, test_size=args.test_size, cat_iterations=args.cat_iterations, cat_random_state=args.seed, 
                                            k_fold=args.k_fold, random_state=args.seed, batch_size=args.batch_size_loader, shuffle=True, class_num=args.class_num, run_dir=run_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    n_features = env.n_features

    agents: List[DQNAgent] = [DQNAgent(state_dim=n_features, hidden=args.hidden, lr=args.lr, buffer_capacity=args.buffer_capacity, 
                                       target_update_every=args.target_update_every, gamma=args.gamma)for _ in range(n_features)]
    eps_sched = LinearEpsilon(start=args.eps_start, end=args.eps_end, decay_episodes=args.eps_decay)
    
    # Logging
    best_acc, best_mask = -1.0, None
    per_agent_logs: List[list] = [[] for _ in range(n_features)]
    
    
    print_log(logger, f"[INFO] Dataset ready. n_features={n_features}")
    print_log(logger, f"[INFO] Dataset features: {env.feature_names}\n")
    
    print_log(logger, "Arguments")
    kv = vars(args)
    width = max(len(k) for k in kv) if kv else 0
    for k in iter(kv):
        print_log(logger, f"  {k:<{width}} : {kv[k]}")
    print_log(logger, " ")
    
    A_t_prev = None

    for ep in range(1, args.episodes + 1):
        epsilon = eps_sched.value(ep)

        # 1) The mask A_t is constructed as the agents sequentially select actions within an episode.
        states_seen = []
        actions = []
        A_t = np.zeros(n_features, dtype=np.int64)
        
        for i in range(n_features):
            s_i = A_t.copy()
            states_seen.append(s_i.astype(np.float32))
            a_i = agents[i].select_action(s_i.astype(np.float32), epsilon=epsilon, device=device)
            actions.append(a_i)
            A_t[i] = a_i

        # 2) Impartial reward
        rewards, F_t, F_delta = impartial_rewards_accuracy(env, A_t_prev, A_t, args.penalty)

        # 3) Save transition (End of episode, terminal reward given)
        for i in range(n_features):
            t = Transition(s=states_seen[i], a=int(actions[i]), r=float(rewards[i]), sp=states_seen[i], done=True)
            agents[i].push_transition(t)

        last_losses = []
        for ag in agents:
            last_losses.append(ag.learn(batch_size=args.batch_size, device=device))

        # 4) Logging (Selected features, Q-value, loss, accuracy)
        selected_features = np.where(A_t == 1)[0].tolist()
        if ep % max(1, args.log_every) == 0:
            print_log(logger, f"[EP {ep:04d}] Selected features: {selected_features}")

        if F_t > best_acc:
            best_acc, best_mask = F_t, A_t.copy()

        # Q(s,0/1)
        for i in range(n_features):
            q0, q1 = agents[i].q_values(states_seen[i], device=device)
            row = {"episode": ep, "epsilon": epsilon, "loss": (None if last_losses[i] is None else float(last_losses[i])), "q0": q0, "q1": q1, "reward": float(rewards[i]), "rf_val_acc": float(F_t)}
            per_agent_logs[i].append(row)

        if ep % max(1, args.log_every) == 0:
            valid_losses = [x for x in last_losses if x is not None]
            avg_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")
            print_log(logger, f"[EP {ep:04d}] eps={epsilon:.3f} acc={F_t:.4f} (\u0394={F_delta:+.4f}) | best={best_acc:.4f} | avg_loss(sample)={avg_loss:.6f}")

        # 5) For the next episode: keep the current mask as prev
        A_t_prev = A_t.copy()
        
    # Save CSVs and plots per agent
    summary = save_plots(run_dir, n_features, per_agent_logs, best_acc, best_mask)
    print_log(logger, " ")
    for k, v in summary.items():
        print_log(logger, f"{k}: {v}")


def parse_args():
    p = argparse.ArgumentParser()
    # DDQN
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for DDQN learning")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer_capacity", type=int, default=50000)
    p.add_argument("--target_update_every", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)

    # Epsilon
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=200)
    
    # Reward
    p.add_argument("--penalty", type=float, default=0.1, help="Penalty for using a feature")
    
    # Dataset
    p.add_argument("--dataset", type=str, default="breast_cancer", choices=["breast_cancer", "sarcopenia"])
    p.add_argument("--sarc_data_path", type=str, default="./data/post-ref4.sav")
    p.add_argument("--k_fold", type=int, default=0, help="<=1: hold-out, >=2: StratifiedKFold")
    p.add_argument("--batch_size_loader", type=int, default=32, help="SarcoDataLoader batch size")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--class_num", type=int, default=2)
    
    # CatBoost
    p.add_argument("--cat_iterations", type=int, default=100)

    # logging
    p.add_argument("--log_every", type=int, default=10)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device)
