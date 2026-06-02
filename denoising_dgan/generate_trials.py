import json
import random

NUM_TRIALS = 10

LR_G_VALUES = [5e-5, 1e-4, 1.5e-4, 2e-4]
LR_D_VALUES = [5e-5, 1e-4, 1.5e-4]
R1_VALUES = [0.01, 0.05, 0.1]
TIMESTEP_VALUES = [2, 4]

trials = {}

for trial_id in range(NUM_TRIALS):

    trials[str(trial_id)] = {
        "seed": random.randint(0, 999999),
        "lr_g": random.choice(LR_G_VALUES),
        "lr_d": random.choice(LR_D_VALUES),
        "r1_gamma": random.choice(R1_VALUES),
        "num_timesteps": random.choice(TIMESTEP_VALUES)
    }

with open("trial_configs.json", "w") as f:
    json.dump(trials, f, indent=4)

print("Saved trial_configs.json")