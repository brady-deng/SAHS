import seaborn as sns
import numpy as np
flights = sns.load_dataset("flights")
flights = flights.pivot("month","year","passengers")
ax = sns.heatmap(flights)