import numpy as np

# Simulation settings
T = 100.0             # total observation time
num_trials = 1000000   # number of simulation runs

# Define different lambda values
lambda_A = 1.0
lambda_B = 2.0
lambda_C = 3.0

# Counters
first_A_count = 0
last_A_count  = 0
num_nonempty  = 0  # how many trials had ≥1 total event

for _ in range(num_trials):
    # 1) Generate the Poisson counts in [0, T]
    n_A = np.random.poisson(lambda_A * T)
    n_B = np.random.poisson(lambda_B * T)
    n_C = np.random.poisson(lambda_C * T)

    # 2) If each count is zero, there are no events at all in this trial
    if (n_A + n_B + n_C) == 0:
        continue

    # 3) Otherwise, we have at least one event in [0, T]
    num_nonempty += 1

    # 4) Generate the actual event times, uniform on [0, T]
    times_A = np.random.uniform(0, T, n_A) if n_A > 0 else np.array([])
    times_B = np.random.uniform(0, T, n_B) if n_B > 0 else np.array([])
    times_C = np.random.uniform(0, T, n_C) if n_C > 0 else np.array([])

    # 5) Tag and combine them
    all_events = []
    all_events += [(t, 'A') for t in times_A]
    all_events += [(t, 'B') for t in times_B]
    all_events += [(t, 'C') for t in times_C]

    # 6) Sort by time to find the first/last
    all_events.sort(key=lambda x: x[0])
    first_label = all_events[0][1]
    last_label  = all_events[-1][1]

    # 7) Tally if A was first or last
    if first_label == 'A':
        first_A_count += 1
    if last_label == 'A':
        last_A_count += 1

# 8) Compute the conditional estimates (divide by num_nonempty, not num_trials)
first_A_prob = first_A_count / num_nonempty
last_A_prob  = last_A_count  / num_nonempty

print("Trials with ≥1 event:", num_nonempty)
print("Estimated P[first=A|≥1 event] =", first_A_prob)
print("Estimated P[last=A|≥1 event]  =", last_A_prob)
