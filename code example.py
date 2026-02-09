# example of how I cleaned the data
# README for the dataset - https://data.world/databeats/college-completion/workspace/file?filename=README.txt
import pandas as pd

grad_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
# the encoding part here is important to properly read the data! It doesn't apply to ALL csv files read from the web,
# but it was necessary here.
grad_data.info()

#%%
# We have a lot of data! A lot of these have many missing values or are otherwise not useful.
to_drop = list(range(39, 56))
to_drop.extend([27, 9, 10, 11, 28, 36, 60, 56])
#%%
grad_data1 = grad_data.drop(grad_data.columns[to_drop], axis=1)
grad_data1.info()
#%%
# drop even more data that doesn't look predictive
drop_more = [0,2,3,6,8,11,12,14,15,18,21,23,29,32,33,34,35]
grad_data2 = grad_data1.drop(grad_data1.columns[drop_more], axis=1)
grad_data2.info()
#%%
print(grad_data2.head())
#%%
import numpy as np
grad_data2.replace('NULL', np.nan, inplace=True)
#%%
grad_data2['hbcu'] = [1 if grad_data2['hbcu'][i]=='X' else 0 for i in range(len(grad_data2['hbcu']))]
grad_data2['hbcu'].value_counts()
#%%
grad_data2['hbcu'] = grad_data2.hbcu.astype('category')
# convert more variables to factors
grad_data2[['level', 'control']] = grad_data2[['level', 'control']].astype('category')
#%%
# In R, we convert vals to numbers, but they already are in this import
grad_data2.info()
#%%
# check missing data
import seaborn as sns

sns.displot(
    data=grad_data2.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
#%%
#let's drop med_stat_value then delete the rest of the NA rows
grad_data2 = grad_data2.drop(grad_data[['med_sat_value']], axis=1)
grad_data2.dropna(axis = 0, how = 'any', inplace = True)
#%%
sns.displot(
    data=grad_data2.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
