import streamlit as st
import pandas as pd
import numpy as np

st.title('Streamlit Demo')


###

st.subheader('SB SW data test:')

# SB data tests

filename = 'CEREStest.csv'
data = pd.read_csv( filename )
df = data.fillna(0) 

if st.checkbox('Show raw SB data'):
    st.subheader('Raw data')
    st.write(df)

# plot test

filename = 'pos_vel.csv'
df = pd.read_csv( filename )

#timeplot = df [ ['Time', 'Long', 'Lat'] ]
tdf = df[ ['Time', ' Long', ' Lat'] ].copy()

print("tdf: ")
tdf.head()

st.line_chart(tdf, x='Time')


# plot test 2

filename = 'albedo_press.csv'
df = pd.read_csv( filename )

#timeplot = df [ ['Time', 'Long', 'Lat'] ]
tdf = df[ ['Time', ' AMAG'] ].copy()

#print("tdf: albedo ")
#tdf.head()

albedovals =  tdf[[' AMAG']]


irfilename = 'iralbedo_press.csv'
print("file 2: ", irfilename) 
irdf = pd.read_csv( irfilename )
irdf = irdf.fillna(0) 
#print(irdf.head())

#irvals = irdf[[' AMAG']] # make a column - series?
#print('irvals:', irvals)

#second df, can rename too?
df2 = pd.DataFrame().assign( Time=irdf['Time'], IRAmag = irdf[' AMAG'])

#print("df2: ")
#print(df2.head())

df3 = df2.assign( Amag=albedovals )

#print("df3: combines irdf and tdf ")
#print(df3.head())


st.line_chart(df3, x='Time')




'''
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
data = load_data(10000)

# Notify the reader that the data was successfully loaded.
#data_load_state.text('Loading data...done!')
data_load_state.text('Done! (using st.cache)')


### Data table

#st.subheader('Raw data')
#st.write(data)

## Replace data with option checkbox:
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


### Histogram

st.subheader('Number of pickups by hour')

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)


### Plot data on a map

st.subheader('Map of all pickups')
st.map(data)

## Redraw maps to show pickups at 17:00

st.subheader(f'Map of all pickups by hour:')
#hour_to_filter = 17
hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h

filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

'''