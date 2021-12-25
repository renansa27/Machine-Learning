# Upper Confidence Bound (UCB)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
users_numb = 10000
ads_numb = 10
ads_selected = []
numbers_of_selections = [0] * ads_numb # = [0,0,0,0,..,0] n times
sums_of_rewards = [0] * ads_numb
total_reward = 0
for n in range(0, users_numb):
    ad = 0
    max_upper_bound = 0
    for i in range(0,ads_numb):
        #If the customer/user clicked on this ad
        if numbers_of_selections[i] > 0:
            avg_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            #It is a high value
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()