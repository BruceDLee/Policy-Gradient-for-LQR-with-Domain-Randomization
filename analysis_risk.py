import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
with open('stabilization_results_risk_10.pkl', 'rb') as f:
  results = pickle.load(f)

all_samples_Ks = results['all_samples_Ks']
all_samples_SA_cost = results['all_samples_SA_cost']
all_samples_DR_cost = results['all_samples_DR_cost']

all_Ks_10 = all_samples_Ks[0]
all_SA_cost_10 = all_samples_SA_cost[0]
all_DR_cost_10 = all_samples_DR_cost[0]

all_Ks_20 = all_samples_Ks[1]
all_SA_cost_20 = all_samples_SA_cost[1]
all_DR_cost_20 = all_samples_DR_cost[1]

all_Ks_500 = all_samples_Ks[2]
all_SA_cost_500 = all_samples_SA_cost[2]
all_DR_cost_500 = all_samples_DR_cost[2]


several_files = False

if several_files:
    with open('stabilization_results_11-50.pkl', 'rb') as f:
        results_11_50 = pickle.load(f)

    all_samples_Ks_11_50 = results_11_50['all_samples_Ks']
    all_samples_SA_cost_11_50 = results_11_50['all_samples_SA_cost']
    all_samples_DR_cost_11_50 = results_11_50['all_samples_DR_cost']

    all_Ks_10 = all_samples_Ks[0] + all_samples_Ks_11_50[0]
    all_Ks_20 = all_samples_Ks[1] + all_samples_Ks_11_50[1]
    all_Ks_500 = all_samples_Ks[2] + all_samples_Ks_11_50[2]
    all_samples_Ks = [all_Ks_10, all_Ks_20, all_Ks_500]

    # SA error  
    all_SA_cost_10 = all_samples_SA_cost[0] + all_samples_SA_cost_11_50[0]
    all_SA_cost_20 = all_samples_SA_cost[1] + all_samples_SA_cost_11_50[1]
    all_SA_cost_500 = all_samples_SA_cost[2] + all_samples_SA_cost_11_50[2]
    all_samples_SA_cost = [all_SA_cost_10, all_SA_cost_20, all_SA_cost_500]

    # DR cost
    all_DR_cost_10 = all_samples_DR_cost[0] + all_samples_DR_cost_11_50[0]
    all_DR_cost_20 = all_samples_DR_cost[1] + all_samples_DR_cost_11_50[1]
    all_DR_cost_500 = all_samples_DR_cost[2] + all_samples_DR_cost_11_50[2]
    all_samples_DR_cost = [all_DR_cost_10, all_DR_cost_20, all_DR_cost_500]


    # save the results
    with open(f'stabilization_results_1-100.pkl', 'wb') as f:
        pickle.dump({
            'all_samples_Ks': all_samples_Ks,
            'all_samples_SA_cost': all_samples_SA_cost,
            'all_samples_DR_cost': all_samples_DR_cost
        }, f)


max_iter = len(all_Ks_10[0])

for i in range(len(all_samples_Ks)):
    for j in range(len(all_samples_Ks[i])):
        if len(all_samples_Ks[i][j]) > max_iter:
            max_iter = len(all_samples_Ks[i][j])

all_Ks_10_errors = np.zeros((len(all_Ks_10), max_iter))
all_Ks_20_errors = np.zeros((len(all_Ks_20), max_iter))
all_Ks_500_errors = np.zeros((len(all_Ks_500), max_iter))

for i in range(len(all_Ks_10)):
    all_Ks_10_errors[i, :len(all_Ks_10[i])] = la.norm(np.nan_to_num((np.array(all_Ks_10[i][:]) - np.array(all_Ks_10[i][-1:])).squeeze(), nan=1e6, posinf=1e6, neginf=-1e6), axis=1)
    if len(all_Ks_10[i]) < max_iter:
        all_Ks_10_errors[i, len(all_Ks_10[i]):] = all_Ks_10_errors[i, len(all_Ks_10[i])-1]
    all_Ks_20_errors[i, :len(all_Ks_20[i])] = la.norm(np.nan_to_num((np.array(all_Ks_20[i][:]) - np.array(all_Ks_20[i][-1:])).squeeze(), nan=1e6, posinf=1e6, neginf=-1e6), axis=1)
    if len(all_Ks_20[i]) < max_iter:
        all_Ks_20_errors[i, len(all_Ks_20[i]):] = all_Ks_20_errors[i, len(all_Ks_20[i])-1]
    all_Ks_500_errors[i, :len(all_Ks_500[i])] = la.norm(np.nan_to_num((np.array(all_Ks_500[i][:]) - np.array(all_Ks_500[i][-1:])).squeeze(), nan=1e6, posinf=1e6, neginf=-1e6), axis=1)
    if len(all_Ks_500[i]) < max_iter:
        all_Ks_500_errors[i, len(all_Ks_500[i]):] = all_Ks_500_errors[i, len(all_Ks_500[i])-1]



# initialize the errors
all_Jsa_10_errors = np.zeros((len(all_samples_SA_cost[0]), max_iter))
all_Jsa_20_errors = np.zeros((len(all_samples_SA_cost[1]), max_iter))
all_Jsa_500_errors = np.zeros((len(all_samples_SA_cost[2]), max_iter))  

# calculate the errors
for i in range(len(all_samples_SA_cost[0])):
    all_Jsa_10_errors[i, :len(all_SA_cost_10[i])] = np.clip(np.array(all_SA_cost_10[i][:]) - np.array(all_SA_cost_10[i][-1:]), 0, 125)
    if len(all_SA_cost_10[i]) < max_iter:
        all_Jsa_10_errors[i, len(all_SA_cost_10[i]):] = all_Jsa_10_errors[i, len(all_SA_cost_10[i])-1]
    all_Jsa_20_errors[i, :len(all_SA_cost_20[i])] = np.clip(np.array(all_SA_cost_20[i][:]) - np.array(all_SA_cost_20[i][-1:]), 0, 125)
    if len(all_SA_cost_20[i]) < max_iter:
        all_Jsa_20_errors[i, len(all_SA_cost_20[i]):] = all_Jsa_20_errors[i, len(all_SA_cost_20[i])-1]
    all_Jsa_500_errors[i, :len(all_SA_cost_500[i])] = np.clip(np.array(all_SA_cost_500[i][:]) - np.array(all_SA_cost_500[i][-1:]), 0, 125)
    if len(all_SA_cost_500[i]) < max_iter:
        all_Jsa_500_errors[i, len(all_SA_cost_500[i]):] = all_Jsa_500_errors[i, len(all_SA_cost_500[i])-1]


all_Jdr_10_errors = np.zeros((len(all_samples_DR_cost[0]), max_iter))   
all_Jdr_20_errors = np.zeros((len(all_samples_DR_cost[1]), max_iter))
all_Jdr_500_errors = np.zeros((len(all_samples_DR_cost[2]), max_iter))

for i in range(len(all_samples_DR_cost[0])):
    all_Jdr_10_errors[i, :len(all_DR_cost_10[i])] = np.array(all_DR_cost_10[i][:])
    if len(all_DR_cost_10[i]) < max_iter:
        all_Jdr_10_errors[i, len(all_DR_cost_10[i]):] = all_Jdr_10_errors[i, len(all_DR_cost_10[i])-1]
    all_Jdr_20_errors[i, :len(all_DR_cost_20[i])] = np.array(all_DR_cost_20[i][:])
    if len(all_DR_cost_20[i]) < max_iter:
        all_Jdr_20_errors[i, len(all_DR_cost_20[i]):] = all_Jdr_20_errors[i, len(all_DR_cost_20[i])-1]
    all_Jdr_500_errors[i, :len(all_DR_cost_500[i])] = np.array(all_DR_cost_500[i][:])
    if len(all_DR_cost_500[i]) < max_iter:
        all_Jdr_500_errors[i, len(all_DR_cost_500[i]):] = all_Jdr_500_errors[i, len(all_DR_cost_500[i])-1]

# display 3 plots in one figure
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(np.median(all_Ks_10_errors, axis=0), label='M=10')
# plot the 25-75% quantile
plt.fill_between(np.arange(max_iter),
                    np.percentile(all_Ks_10_errors, 75, axis=0),
                    np.percentile(all_Ks_10_errors, 25, axis=0),
                    alpha=0.3)

plt.plot(np.median(all_Ks_20_errors, axis=0), label='M=20')
# plot the 25-75% quantile
plt.fill_between(np.arange(max_iter),
                 np.percentile(all_Ks_20_errors, 75, axis=0),
                 np.percentile(all_Ks_20_errors, 25, axis=0),
                 alpha=0.3)

plt.plot(np.median(all_Ks_500_errors, axis=0), label='M=500')
# plot the 25-75% quantile
plt.fill_between(np.arange(max_iter),
                 np.percentile(all_Ks_500_errors, 75, axis=0),
                 np.percentile(all_Ks_500_errors, 25, axis=0),
                 alpha=0.3)

plt.grid()
plt.legend(fontsize=14)
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel(r'$\|K-K_{SA}^\star\|^2$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 30))

plt.subplot(1, 3, 2)

plt.plot(np.median(all_Jsa_10_errors, axis=0), label='M=10')
plt.fill_between(np.arange(len(all_Jsa_10_errors[1])),
                 np.percentile(all_Jsa_10_errors, 75, axis=0),
                 np.percentile(all_Jsa_10_errors, 25, axis=0),
                 alpha=0.3)

plt.plot(np.median(all_Jsa_20_errors, axis=0), label='M=20')
plt.fill_between(np.arange(len(all_Jsa_20_errors[1])),
                 np.percentile(all_Jsa_20_errors, 75, axis=0),
                 np.percentile(all_Jsa_20_errors, 25, axis=0),
                 alpha=0.3)

plt.plot(np.median(all_Jsa_500_errors, axis=0), label='M=500')
plt.fill_between(np.arange(len(all_Jsa_500_errors[1])),
                 np.percentile(all_Jsa_500_errors, 75, axis=0),
                 np.percentile(all_Jsa_500_errors, 25, axis=0),
                 alpha=0.3)

# plt.ylim((0.5*10**-7, 10**3))
plt.grid()
plt.legend(fontsize=14)
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel(r'$J_{SA}(K)-J_{SA}(K_{SA}^\star)$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(1, 3, 3)

plt.plot(np.median(all_Jdr_10_errors, axis=0), label='M=10')
# plot the 25-75% quantile
plt.fill_between(np.arange(len(all_Jdr_10_errors[1])),
                 np.percentile(all_Jdr_10_errors, 75, axis=0),
                 np.percentile(all_Jdr_10_errors, 25, axis=0),
                 alpha=0.3)

plt.plot(np.median(all_Jdr_20_errors, axis=0), label='M=20')
# plot the 25-75% quantile
plt.fill_between(np.arange(len(all_Jdr_20_errors[1])),
                 np.percentile(all_Jdr_20_errors, 75, axis=0),
                 np.percentile(all_Jdr_20_errors, 25, axis=0),
                 alpha=0.3)

plt.plot(np.median(all_Jdr_500_errors, axis=0), label='M=500')
# plot the 25-75% quantile
plt.fill_between(np.arange(len(all_Jdr_500_errors[1])),
                 np.percentile(all_Jdr_500_errors, 75, axis=0),
                 np.percentile(all_Jdr_500_errors, 25, axis=0),
                 alpha=0.3)


plt.ylim((7800, 10000))
plt.grid()
plt.legend(fontsize=14)
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel(r'$J_{DR}(K)$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig(f'stabilization_results_risk_{len(all_Ks_10)}.png', dpi=300)
plt.show()