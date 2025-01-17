import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sim = pd.DataFrame()

iterations = 5000

expected_return = 0.09
volatility = 0.18
time_horizion = 30
initial_value = 10000
annual_investement = 10000

for x in range(iterations):
    pv = initial_value
    path = []

    for y in range(time_horizion):
        market_return = np.random.normal(expected_return, volatility)
        pv = pv * (1 + market_return) + annual_investement

        path.append(pv)
    
    sim[x] = path

#Summary Statistics
ending_values = sim.loc[29]
mean = np.mean(ending_values)
std = np.std(ending_values)
min = np.min(ending_values)
max = np.max(ending_values)

print("Summary Statistcs For a Simulation of n = 5000")
print("Mean: $" + str("{:,}".format(round(mean, 2))))
print("Standard Deviation: $" + str("{:,}".format(round(std, 2))))
print("Min: $" + str("{:,}".format(round(min, 2))))
print("Max: $" + str("{:,}".format(round(max, 2))))



#probability final amount is less than 1mil
prob_below_1m = (ending_values < 1_000_000).sum() / len(ending_values)
prob_at_least_1m = 1 - prob_below_1m
percentiles = np.percentile(ending_values, [5, 25, 75])

print("Probability Estimation and Percentiles")
print("Probability of less than 1m: " + str(round(100 * prob_below_1m,2)) + "%")
print("Probability of at least than 1m: " + str(round(100 * prob_at_least_1m,2)) + "%")
print("5th percentile: " + str("{:,}".format(round(percentiles[0], 2))))
print("25th percentile: " + str("{:,}".format(round(percentiles[1], 2))))
print("75th percentile: " + str("{:,}".format(round(percentiles[2], 2))))



#graph plot
plt.figure(figsize=(10, 6))
plt.hist(ending_values, bins=100, color='skyblue', edgecolor='black')
 
# Add labels, title, and reference line
plt.title("Distribution of Final Year Balances After 30 Years", fontsize=14)
plt.xlabel("Ending Balance ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label="Mean: $" + str("{:,}".format(round(mean, 2))))
plt.axvline(percentiles[0], color='orange', linestyle='dashed', linewidth=1.5, label="5th Percentile: $" + str("{:,}".format(round(percentiles[0], 2))))
plt.axvline(percentiles[1], color='green', linestyle='dashed', linewidth=1.5, label="25th Percentile: $" + str("{:,}".format(round(percentiles[1], 2))))
plt.axvline(percentiles[2], color='purple', linestyle='dashed', linewidth=1.5, label="75th Percentile: $" + str("{:,}".format(round(percentiles[2], 2))))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.show()