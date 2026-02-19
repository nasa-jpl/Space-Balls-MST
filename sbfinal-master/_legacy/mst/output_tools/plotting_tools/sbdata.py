# SB data tests
import pandas as pd
import numpy as np

filename = 'CEREStest.csv'
data = pd.read_csv( filename )
df = data.fillna(0) 

#if st.checkbox('Show raw SB data'):
#    st.subheader('Raw data')
#    st.write(df)

# plot test

filename = 'pos_vel.csv'
df = pd.read_csv( filename )
print(df.head())

print("")

mylist = list(df)
print('mylist pos-vel: ', mylist)

#timeplot = df [ ['Time', 'Long', 'Lat'] ]
tdf = df[ ['Time', ' Long', ' Lat'] ].copy()

print("tdf df copy: ")
print(tdf.head())


filename = 'albedo_press.csv'
print("loading: ", filename)
df = pd.read_csv( filename )
df = df.fillna(0) 
print(df.head())

print("")

mylist = list(df)
print('mylist albedo: ', mylist)
'''
#timeplot = df [ ['Time', 'Long', 'Lat'] ]

tdf = df[ ['Time', ' Long', ' Lat'] ].copy()

print("tdf: ")
print(tdf.head())
'''

albedovals = df[[' AMAG']]


# get column from another file, put in another dataframe

filename = 'iralbedo_press.csv'
print("file 2: ", filename) 
irdf = pd.read_csv( filename )
irdf = irdf.fillna(0) 
print(irdf.head())

irvals = irdf[[' AMAG']] # make a column - series?

print('irvals:', irvals)

#second df, can rename too?
df2 = pd.DataFrame().assign( Time=irdf['Time'], IRAmag = irdf[' AMAG'])


print("df2: ")
print(df2.head())

# combine 2 dataframe columns into new dataframe:

#df3 =pd.concat( [irdf,df2], axis=0, ignore_index=True )

#df3 = pd.merge(irdf,df2)

df3 = df2.assign( Amag=albedovals )

print("df3: combines irdf and df2 ")
print(df3.head())







