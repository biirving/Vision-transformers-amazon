import pandas as pd
from glob import glob

"""
For sorting through the created dataframes
"""


dataframes = []

for csv in glob("beit*.csv"):
    dataframes.append(pd.read_csv(csv))
    print(csv)

aggr_df = pd.concat(dataframes)
aggr_df

# Query best latency
#aggr_df.sort_values(by='latency_ms_p90', ascending=True)[0:3]

# Query best throughput
#aggr_df.sort_values(by='throughput average', ascending=False)[0:3]


# Query best price
#aggr_df.sort_values(by='cost per 1 m instances', ascending=False)[0:3]


# Lowest p90 latency
lowest_cost = aggr_df.sort_values(by='cost_per_1m_inf', ascending=True)[0:10]
lowest_cost.to_csv('beit_lowest_cost.csv')

highest_throughput = aggr_df.sort_values(by='throughput_avg', ascending=False)[0:10]
highest_throughput.to_csv('beit_highest_throughput.csv')

lowest_latency = aggr_df.sort_values(by='latency_ms_p90', ascending=True)[0:10]
lowest_latency.to_csv('beit_lowest_latency.csv')


# Cheapest Price
#print(aggr_df.sort_values(by='cost per 1 m instances', ascending=True)[0:3])

# Highest average throughput
#print(aggr_df.sort_values(by='throughput average', ascending=False)[0:3])