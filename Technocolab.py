import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

st.write("""
# Spotify Sequential Skip Prediction
## - by Chandraprakash Koshle : [chandraprakash.iitkgp@gmail.com] (chandraprakash.iitkgp@gmail.com)

This app predicts wheather a track played in spotify will be **skipped** by the user or not """)
st.sidebar.header("User Input track and it's feature data ")

st.sidebar.markdown("""
[Sample CSV input file](https://github.com/cp2611/Technocolabs-Internship-Project/blob/master/sample_input.csv)
""")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        premium = st.sidebar.selectbox('premium',(0,1))
        start_reason = st.sidebar.selectbox('How did you get to this track ?',("start_appload","start_backbtn","start_clickrow","start_endplay","start_fwdbtn","start_playbtn","start_remote","start_trackdone","start_trackerror"))
        end_reason = st.sidebar.selectbox('How did you end the tracks on Spotify on an average ?',("end_appload","end_backbtn","end_clickrow","end_endplay","end_fwdbtn","end_logout","end_remote","end_trackdone","end_trackerror"))
        session_position = st.sidebar.slider('session_position', 0,12,5)/12
        context_switch = st.sidebar.selectbox('context switch',(0,1))
        hour_of_the_day=st.sidebar.slider('session_position', 0.0,1.0,0.5)
        hist_user_behavior_n_seekfwd=-0.0023
        hist_user_behavior_n_seekback=-0.023
        pause_type=st.sidebar.selectbox('what kind of pause you take before the play very often ?',("no_pause","short_pause","long_pause"))
        hist_user_behavior_is_shuffle=st.sidebar.selectbox('Do you like to shuffle before play',(0,1))
        context_type=st.sidebar.selectbox('type of the context', ("catalog","charts","editorial_playlist","personalised_playlist","radio","user_collection"))

        behavior = {"start_appload":0,
                    "start_backbtn":0,
                    "start_clickrow":0,
                    "start_endplay":0,
                    "start_fwdbtn":0,
                    "start_playbtn":0,
                    "start_remote":0,
                    "start_trackdone":0,
                    "start_trackerror":0,
                    "end_appload":0,
                    "end_backbtn":0,
                    "end_clickrow":0,
                    "end_endplay":0,
                    "end_fwdbtn":0,
                    "end_logout":0,
                    "end_remote":0,
                    "end_trackdone":0,
                    "no_pause":0,
                    "short_pause":0,
                    "long_pause":0,
                    "catalog":0,
                    "charts":0,
                    "editorial_playlist":0,
                    "personalised_playlist":0,
                    "radio":0,
                    "user_collection":0

                    }

        behavior.update({behavior[start_reason]:1,
                    behavior[end_reason]:1,
                    behavior[pause_type]:1,
                    behavior[context_type]:1})
        duration=st.sidebar.slider("Track Duration",1,10,5)/10*(19.2+2.3)-2.3
        release_year=st.sidebar.slider("How new is the track ?",0.0,1.0,0.9)

        acousticness=	st.sidebar.slider ("acousticness",0.0,1.0,0.7)
        beat_strength	=st.sidebar.slider("beat strength",0.0,1.0,0.7)
        bounciness	=st.sidebar.slider("bounciness",0.0,1.0,0.7)
        danceability	=st.sidebar.slider("danceability",0.0,1.0,0.7)
        dyn_range_mean	=st.sidebar.slider("dyne_range mean",0.0,1.0,0.7)*(6.4+3.24)-3.24
        energy	=st.sidebar.slider("energy",0.0,1.0,0.7)
        flatness	=st.sidebar.slider("flatness",0.0,1.0,0.7)*1.13
        instrumentalness	=st.sidebar.slider("instrumentalness",0.0,1.0,0.7)
        key	=st.sidebar.slider("key",0.0,1.0,0.7)
        liveness	=st.sidebar.slider("liveness",0.0,1.0,0.7)
        loudness	=st.sidebar.slider("loudness",0.0,1.0,0.7)*(13.66)-11.66
        mechanism	=st.sidebar.slider("mechanism",0.0,1.0,0.7)
        is_major	=st.sidebar.selectbox("major or not",(0,1))
        organism	=st.sidebar.slider("organism",0.0,1.0,0.7)
        speechiness	=st.sidebar.slider("speechness",0.0,1.0,0.7)
        tempo	=st.sidebar.slider("tempo",0.0,1.0,0.7)*7-3
        time_signature	=st.sidebar.selectbox("time signature",(0,1,2,3,4,5))*1.4
        valence	=st.sidebar.slider("valence",0.0,1.0,0.7)
        acoustic_vector_0	=st.sidebar.slider("acoustic_vector_0",0.0,1.0,0.7)*(2.74+2.06)-2.06
        acoustic_vector_1	=st.sidebar.slider("acoustic_vector_1",0.0,1.0,0.7)*(2+4.2)-4.2
        acoustic_vector_2	=st.sidebar.slider("acoustic_vector_2",0.0,1.0,0.7)*(1.8+3.7)-3.7
        acoustic_vector_3	=st.sidebar.slider("acoustic_vector_3",0.0,1.0,0.7)*(2.9+2.3)-2.3
        acoustic_vector_4	=st.sidebar.slider("acoustic_vector_4",0.0,1.0,0.7)*(2.44+2.65)-2.65
        acoustic_vector_5	=st.sidebar.slider("acoustic_vector_5",0.0,1.0,0.7)*(2.14+6.13)-6.28
        acoustic_vector_6=st.sidebar.slider("acoustic_vector_6",0.0,1.0,0.7)*(3.13+1.17)-1.17
        acoustic_vector_7=st.sidebar.slider("acoustic_vector_7",0.0,1.0,0.7)*(3.44+2.88)-2.88



        data = {'premium': premium,
                'acoustic_vector_7':acoustic_vector_7,
                'acoustic_vector_6':acoustic_vector_6,
                'acoustic_vector_5':acoustic_vector_5,
                'acoustic_vector_4':acoustic_vector_4,
                'acoustic_vector_3':acoustic_vector_3,
                'acoustic_vector_2':acoustic_vector_2,
                'acoustic_vector_1':acoustic_vector_1,
                'acoustic_vector_0':acoustic_vector_0,
                'duration':duration,
                'release_year':release_year,
                'us_popularity_estimate' :0.02,
                'acousticness':acousticness,
                'beat_strength':beat_strength,
                'bounciness':bounciness,
                'danceability':danceability,
                'dyn_range_mean':dyn_range_mean,
                'energy':energy,	
                'flatness':flatness,
                'instrumentalness':instrumentalness,
                'key'	:key,
                'liveness':liveness,
                'loudness':loudness,
                'mechanism':mechanism,
                'is_major':is_major,
                'organism':organism,
                'speechiness':speechiness,
                'tempo':tempo,
                'time_signature':time_signature,
                'valence':valence  , 
                'session_position':session_position,	
                'session_length'	:1,	
                'context_switch':context_switch,	
                'no_pause_before_play':behavior["no_pause"],	
                'short_pause_before_play':behavior["short_pause"],	
                'long_pause_before_play':behavior["long_pause"],	
                'hour_of_day':hour_of_the_day,	
                'context_type_catalog':behavior["catalog"],	
                'context_type_charts':behavior["charts"],	
                'context_type_editorial_playlist':behavior["editorial_playlist"],	
                'context_type_personalized_playlist':behavior["personalised_playlist"],	
                'context_type_radio':behavior["radio"],	
                'context_type_user_collection':behavior["user_collection"],	
                'hist_user_behavior_reason_start_appload'	:behavior["start_appload"],
                'hist_user_behavior_reason_start_backbtn':behavior["start_backbtn"],	
                'hist_user_behavior_reason_start_clickrow':behavior["start_clickrow"],	
                'hist_user_behavior_reason_start_endplay'	:behavior["start_endplay"],
                'hist_user_behavior_reason_start_fwdbtn':behavior["start_fwdbtn"],	
                'hist_user_behavior_reason_start_playbtn'	:behavior["start_playbtn"],
                'hist_user_behavior_reason_start_remote'	:behavior["start_remote"],
                'hist_user_behavior_reason_start_trackdone'	:behavior["start_trackdone"],
                'hist_user_behavior_reason_start_trackerror':behavior["start_trackerror"],	
                'hist_user_behavior_reason_end_appload'	:behavior["end_appload"],
                'hist_user_behavior_reason_end_backbtn'	:behavior["end_backbtn"],
                'hist_user_behavior_reason_end_clickrow':behavior["end_clickrow"],
                'hist_user_behavior_reason_end_endplay':behavior["end_endplay"],
                'hist_user_behavior_reason_end_fwdbtn':behavior["end_fwdbtn"],
                'hist_user_behavior_reason_end_logout':behavior["end_logout"],
                'hist_user_behavior_reason_end_remote':behavior["end_remote"],
                'hist_user_behavior_reason_end_trackdone':behavior["end_trackdone"],
                'hist_user_behavior_n_seekfwd':hist_user_behavior_n_seekfwd,
                'hist_user_behavior_n_seekback':hist_user_behavior_n_seekback,
                'hist_user_behavior_is_shuffle' :hist_user_behavior_is_shuffle,
               
                }
        df = pd.DataFrame(data,index=[0])
        return df
    input_df = user_input_features()


st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Reads in saved classification model
model=XGBClassifier({'nthread':4})
model2=XGBClassifier({'nthread':4})
model3=XGBClassifier({'nthread':4})
model.load_model("model_skip_1_xgbooster.booster")
model2.load_model("model_skip_2_xgbooster.booster")
model3.load_model("model_skip_3_xgbooster.booster")
# Apply model to make predictions
features=pickle.load(open("top40features.pkl",'rb'))
pred_skip_1 = np.argmax(model.predict_proba(input_df[features]),axis=1)
pred_skip_2 = np.argmax(model2.predict_proba(input_df[features]),axis=1)
pred_skip_3 = np.argmax(model3.predict_proba(input_df[features]),axis=1)

st.subheader('Predictions')
skip_variety = {'small skip': pred_skip_1,'moderate skip': pred_skip_2,'large skip':pred_skip_3}
if ((pred_skip_1==0)&(pred_skip_2==0)&(pred_skip_3==0)):
  st.write("The track won't be skipped")
else:
  st.write(pd.DataFrame(skip_variety))
