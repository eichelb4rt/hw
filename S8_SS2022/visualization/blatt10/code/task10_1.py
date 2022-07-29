import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("co2_data.csv")

####################
# Task 1a          #
####################
fig, ax1 = plt.subplots()
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.ylim(0,500)

dataframe = dataframe.sort_values(by="Emissions", ascending=False)
top20 = dataframe[:20]
ax1.bar(top20["Country"], top20["Emissions"] / 10**9)
ax1.set_ylabel("Billion Tonnes CO2")









####################
# Task 1b          #
####################
ax2 = ax1.twinx()
plt.ylim(0,1)


ax2.set_ylabel("Cumulative Percentage")
cumsums = np.cumsum(top20["Emissions"]) / dataframe["Emissions"].sum()
ax2.plot(top20["Country"], cumsums, "-ro")











# Show the result
plt.title('Total CO2 Emissions 1751-2017')
plt.tight_layout()
plt.show()