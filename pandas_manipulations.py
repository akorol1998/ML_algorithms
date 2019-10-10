# Sample file with curious code to explore
import pandas as pd
midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],
                              ['speed', 'weight', 'length']],
                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
df = pd.DataFrame(index=midx, columns=['big', 'small'],
                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
                         [250, 150], [1.5, 0.8], [320, 250],
                         [1, 0.8], [0.3, 0.2]])
print(df)
# In this case, if we want to drop some shitt out our dataframe, we can explicitly tell it
df.drop(index='lama', 1)

df.drop(index='length', level=1, inplace=True)
# df.drop(index='cow', columns='small')
df.drop(columns='small', inplace=True)
print(df)