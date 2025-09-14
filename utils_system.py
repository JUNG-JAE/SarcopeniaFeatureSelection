import json
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging


def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_dir(base="./runs"):
    Path(base).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def print_log(logger, msg):
    print(msg)
    logger.info(msg)


def set_logger(run_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    
    filename=f'{run_dir}/logger.log'
    
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def save_plots(run_dir, n_features, per_agent_logs, best_acc, best_mask):    
    csv_paths = []
    for i in range(n_features):
        df = pd.DataFrame(per_agent_logs[i])
        csv_path = run_dir / f"csv/agent_{i:02d}.csv"
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

        # Plots
        window = 100  # moving average window size
        # Loss
        plt.figure()
        plt.plot(df["episode"], df["loss"].rolling(window).mean())
        plt.xlabel("Episode"); plt.ylabel("Loss"); plt.title(f"Agent {i:02d} - Loss per Episode")
        plt.tight_layout()
        plt.savefig(run_dir / "figs" / f"agent_{i:02d}_loss.png")
        plt.close()

        # Q-values
        plt.figure()
        plt.plot(df["episode"], df["q0"].rolling(window).mean(), label="Q(a=0)")
        plt.plot(df["episode"], df["q1"].rolling(window).mean(), label="Q(a=1)")
        plt.xlabel("Episode"); plt.ylabel("Q-value"); plt.title(f"Agent {i:02d} - Q-values per Episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "figs" / f"agent_{i:02d}_q.png")
        plt.close()
        
        # Reward
        plt.figure()
        plt.plot(df["episode"], df["reward"].rolling(window).mean())
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title(f"Agent {i:02d} - Reward per Episode")
        plt.tight_layout()
        plt.savefig(run_dir / "figs" / f"agent_{i:02d}_reward.png")
        plt.close()

    # RF accuracy
    plt.figure()
    plt.plot(df["episode"], df["rf_val_acc"])
    plt.xlabel("Episode"); plt.ylabel("RF Val Accuracy"); plt.title(f"Agent {i:02d} - RF Accuracy per Episode")
    plt.tight_layout()
    plt.savefig(run_dir / "figs" / f"rf_acc.png")
    plt.close()

    # Save best selection summary
    summary = {
        "best_val_acc": float(best_acc),
        "best_subset_size": int(best_mask.sum()) if best_mask is not None else None,
        "best_indices": [int(i) for i in np.where(best_mask == 1)[0]] if best_mask is not None else [],
    }
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary