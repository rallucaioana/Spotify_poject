import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import plotly.express as px


# run using streamlit run Example_streamlit.py


st.write("Hello, let's learn how to build a streeamlit app together")
st.title("This is the app title")
st.header("This is the header")
st.markdown("This is the markdown")
st.subheader("This is the subheader")
st.caption("This is the caption")
st.code("x = 2021")
st.latex(r''' a+a r^1+a r^2+a r^3 ''')
st.checkbox('Yes')
st.button('Click Me')
# gender = st.radio('Pick your gender', ['Male', 'Female'])
# if gender == "Female":
#     st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange','Grapes'])
# if gender == 'Male':
#     st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])

st.write("Album Summary")
album_name = st.text_input(label="Pick an album:")

if(st.button(label="Submit")):
    no_tracks, time_sigs = get_album_data(album_name)
    
    st.write(album_name)
    st.write("Number of tracks", no_tracks)
    
    time_sig_fig = px.pie(data_frame=time_sigs)
    st.plotly_chart(time_sig_fig)
    
   


options =['Jupiter', 'Mars', 'Neptune', 'Earth', 'Venus', 'Pluto', 'Uranus', 'Mercury']
choice = st.selectbox('Choose a planet', options)

st.write("You selected", choice)

Mark = st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
if Mark == 'Bad':
    st.write("Boo!")

if Mark == 'Good':
    st.write("Well done!")

if Mark == 'Excellent':
    st.write("Great!")

nm = st.slider('Pick a number', 0, 50)
st.write("Your number: ", nm)

st.number_input('Pick a number', 0, 10)
st.text_input('Email address')
st.date_input('Traveling date')
st.time_input('School time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')

st.sidebar.title("Sidebar Title")
st.sidebar.markdown("This is the sidebar content")
st.sidebar.button("Click me!")
st.sidebar.radio('Pick your sidebar gender', ['Male', 'Female'])

rand = np.random.normal(1,2,size = 200)
fig, ax = plt.subplots()
ax.hist(rand, bins = 15)
st.pyplot(fig)


df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.line_chart(df)

