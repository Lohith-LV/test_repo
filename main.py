# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:23:33 2023

@author: lohith.boddu
"""

def main():
       with st.form("road_traffic_severity_form"):
              st.subheader("Pleas enter the following inputs:")
              
              No_vehicles = st.slider("Number of vehicles involved:",1,7, value=0, format="%d")
              No_casualties = st.slider("Number of casualities:",1,8, value=0, format="%d")
              Hour = st.slider("Hour of the day:", 0, 23, value=0, format="%d")
              collision = st.selectbox("Type of collision:",options=options_types_collision)
              Age_band = st.selectbox("Driver age group?:", options=options_age)
              Sex = st.selectbox("Sex of the driver:", options=options_sex)
              Education = st.selectbox("Education of driver:",options=options_education_level)
              service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_year)
              Day_week = st.selectbox("Day of the week:", options=options_day)
              Accident_area = st.selectbox("Area of accident:", options=options_acc_area)
              
              submit = st.form_submit_button("Predict")

# encode using ordinal encoder and predict
       if submit:
              input_array = np.array([collision,
                                   Age_band,Sex,Education,service_vehicle,
                                   Day_week,Accident_area], ndmin=2)
              
              encoded_arr = list(encoder.transform(input_array).ravel())
              
              num_arr = [No_vehicles,No_casualties,Hour]
              pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)              
          
# predict the target from all the input features
              prediction = model.predict(pred_arr)
              
              if prediction == 0:
                     st.write(f"The severity prediction is Fatal Injury⚠")
              elif prediction == 1:
                     st.write(f"The severity prediction is serious injury")
              else:
                     st.write(f"The severity prediction is slight injury")
                  
              st.subheader("Explainable AI (XAI) to understand predictions")
# Explainable AI using shap library 
              shap.initjs()
              shap_values = shap.TreeExplainer(model).shap_values(pred_arr)
              st.write(f"For prediction {prediction}") 
              shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0],
                              pred_arr, feature_names=features, matplotlib=True,show=False).savefig("pred_force_plot.jpg", bbox_inches='tight')
              img = Image.open("pred_force_plot.jpg")

# render the shap plot on front-end to explain predictions
              st.image(img, caption='Model explanation using shap')
              
              st.write("Developed By: Avi kumar Talaviya")
              st.markdown("""Reach out to me on: [Twitter](https://twitter.com/avikumart_) |
              [Linkedin](https://www.linkedin.com/in/avi-kumar-talaviya-739153147/) |
              [Kaggle](https://www.kaggle.com/avikumart) 
              """)
              
              a,b,c = st.columns([0.2,0.6,0.2])
              with b:
                  st.image("vllkyt19n98psusds8.jpg", use_column_width=True)


# description about the project and code files            
                  st.subheader("🧾Description:")
                  st.text("""This data set is collected from Addis Ababa Sub-city police departments for master's research work. 
The data set was prepared from manual documents of road traffic accidents of the year 2017-20. 
All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident.
Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.
""")

                  st.markdown("Source of the dataset: [Click Here](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)")

                  st.subheader("🧭 Problem Statement:")
                  st.text("""The target feature is Accident_severity which is a multi-class variable. 
The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. 
The metric for evaluation will be f1-score
""")

                  st.markdown("Please find GitHub repository link of project: [Click Here](https://github.com/avikumart/Road-Traffic-Severity-Classification-Project)")                  
   
# run the main function               
if __name__ == '__main__':
   main()