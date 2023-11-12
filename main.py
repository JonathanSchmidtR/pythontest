import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pycaret 
from pycaret.regression import *
import datetime



data = pd.read_csv("bike-sharing_hourly.csv")

def load_pycaret_model():
    return load_model('final_model')

model = load_pycaret_model()

st.set_page_config(page_title="Washington D.C. Bike-Sharing Analysis", layout="wide")

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "landing"

st.markdown('---')
if st.button('Home', key='home'):
    st.session_state['current_page'] = "landing"
if st.button('Explore Findings', key='findings'):
    st.session_state['current_page'] = "findings"
if st.button('Learn About the Model', key='model'):
    st.session_state['current_page'] = "model"
if st.button('Get Predictions', key='predictions'):
    st.session_state['current_page'] = "predictions"
st.markdown('---')

# Landing Page
if st.session_state['current_page'] == "landing":
    st.title('Washington D.C. Bike-Sharing Analysis')
    st.markdown('---')
    st.header('Welcome to our bike-sharing service analysis and prediction app!')

    # Introduction text
    st.info(
        """
        As part of the consultancy team working with the administration of Washington D.C., we've built this interactive 
        web application to help you gain insights into the city's bike-sharing service and predict bicycle demand.
        """
    )

    # Buttons to navigate to other pages
    if st.button('Explore Findings'):
        st.session_state['current_page'] = "findings"

    if st.button('Learn About the Model'):
        st.session_state['current_page'] = "model"

    if st.button('Get Predictions'):
        st.session_state['current_page'] = "predictions"
    

# Findings Page
elif st.session_state['current_page'] == "findings":
    st.markdown('---')  
    # Content for the Findings page
    st.title('Key Findings')
    st.header('Discover Insights from the Bike-Sharing Service Data')

    # Introduction text for the Findings page
    st.info(
        """
        This page presents the key findings from our analysis of the bike-sharing service data in Washington D.C. 
        These insights can help the transportation department make informed decisions and optimize the service.
        """
    )

    
    casual_sum = data['casual'].sum()
    registered_sum = data['registered'].sum()

    # Pie chart
    labels = 'Casual', 'Registered'
    sizes = [casual_sum, registered_sum]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e., 'Registered')

    fig, ax = plt.subplots(figsize=(2, 2))

    # Create the pie chart
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
         autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Bike Rentals: Casual vs Registered OJO LA LETRA!',fontsize=12)
    st.pyplot(plt)

    
    st.header('Title of the graph')
    st.markdown('comentario de la grafica')

    data['dteday'] = pd.to_datetime(data['dteday'])

    # Calculate total rentals per day
    total_rentals_per_day = data.groupby('dteday')['cnt'].sum()

    # Calculate weekly moving average for 'casual' and 'registered'
    weekly_ma_casual = data.groupby('dteday')['casual'].sum().rolling(window=30).mean()
    weekly_ma_registered = data.groupby('dteday')['registered'].sum().rolling(window=30).mean()

    # Plotting
    plt.figure(figsize=(15, 15))

    # Plot total rentals
    plt.plot(total_rentals_per_day, label='Total Rentals')

    # Plot weekly moving average of 'casual' and 'registered'
    plt.plot(weekly_ma_casual, label='Weekly Moving Avg - Casual', linestyle='--')
    plt.plot(weekly_ma_registered, label='Weekly Moving Avg - Registered', linestyle='--')

    # Title and labels
    plt.title('Bike Rentals Over Time with Weekly Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    st.pyplot(plt)



    st.header('Title of the graph')
    st.markdown('comentario de la grafica')

    plt.figure(figsize=(15, 5))
    sns.regplot(x=data['atemp'] * 100, y='cnt', data=data, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('Scatter Plot with Regression Line of atemp vs. cnt')
    plt.xlabel('Apparent Temperature (atemp) in Farenheit')
    plt.ylabel('Count of Users (cnt)')
    st.pyplot(plt)


# Model Page
elif st.session_state['current_page'] == "model":
    st.markdown('---') 
    # Content for the Model page
    st.title('Comprehensive Model Analysis')
    st.header('Understanding the Bike Demand Prediction Model')

    # Introduction text for the Model page
    st.info(
        """
        This page provides an explanation of the predictive model used for forecasting bicycle demand. 
        We'll describe the model and the reasons for selecting it to optimize provisioning and minimize costs.
        """
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Summary Statistics', 'Data Visualization Insights', 'Outliers', 'Feature Engineering', 'Correlation Insights','Feature Importance Analysis' ])

    # Summary Statistics Section
    with tab1:
        st.markdown("### Summary Statistics")
        st.write(data.describe())  # Display summary statistics
        st.markdown("""
        - Balanced data distribution across years 2011 and 2012, providing a comprehensive temporal view.
        - Minimal influence of holidays on bike rentals, reflecting routine usage patterns.
        - Dominance of working days in the dataset, highlighting the service's role in daily commuting.
        - Wide range in rental numbers, underscoring diverse user behaviors and needs.""")

    with tab2:        
        st.markdown("""
        ### Monthly 
        - Seasonal trends evident, with higher usage in warmer months, indicating weather dependency.""") 
        with st.expander("Chart"):
            st.header('Monthly Bike Rentals')


            # Calculate the percentage for each component
            data['dteday'] = pd.to_datetime(data['dteday'])
            monthly_data = data.groupby(data['dteday'].dt.to_period("M")).agg({'casual': 'sum', 'registered': 'sum'})  
            
            total = monthly_data['casual'] + monthly_data['registered']
            percentage_casual = monthly_data['casual'] / total * 100
            percentage_registered = monthly_data['registered'] / total * 100

            # Create a stacked bar chart
            plt.figure(figsize=(12, 5))
            p1 = plt.bar(monthly_data.index.astype(str), monthly_data['casual'], label='Casual', alpha=0.7)
            p2 = plt.bar(monthly_data.index.astype(str), monthly_data['registered'], bottom=monthly_data['casual'], label='Registered', alpha=0.7)

            # Add percentage labels
            for i, (casual, registered) in enumerate(zip(monthly_data['casual'], monthly_data['registered'])):
                plt.text(i, casual / 2, f'{percentage_casual.iloc[i]:.1f}%', ha='center', va='center', color='white', fontsize=8)
                plt.text(i, casual + registered / 2, f'{percentage_registered.iloc[i]:.1f}%', ha='center', va='center', color='white', fontsize=8)

            plt.title('Monthly Bike Rentals (Casual vs. Registered)')
            plt.ylabel('Number of Users')
            plt.xlabel('Month')

            # Rotate x-axis labels vertically
            plt.xticks(rotation='vertical')

            plt.legend()
            st.pyplot(plt)    

        st.markdown("""
        ### Daily 
        - There is a clear distinction between weekdays and weekends, indicating commuter patterns.""") 
        with st.expander("Chart"):
            st.header('Daily Bike Rentals')
            data['weekday_name'] = data['dteday'].dt.day_name()

            weekday_totals = data.groupby('weekday_name').agg({'casual': 'sum', 'registered': 'sum'}).reset_index()

            # Sort the dataframe by the days of the week assuming the days are not in order
            ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_totals['weekday_name'] = pd.Categorical(weekday_totals['weekday_name'], categories=ordered_days, ordered=True)
            weekday_totals.sort_values('weekday_name', inplace=True)

            # Calculate the percentage for each component
            total = weekday_totals['casual'] + weekday_totals['registered']
            percentage_casual = weekday_totals['casual'] / total * 100
            percentage_registered = weekday_totals['registered'] / total * 100

            # Create the stacked bar plot
            plt.figure(figsize=(10, 5))
            p1 = plt.bar(weekday_totals['weekday_name'], weekday_totals['registered'], label='Registered', alpha=0.7)
            p2 = plt.bar(weekday_totals['weekday_name'], weekday_totals['casual'], bottom=weekday_totals['registered'], label='Casual', alpha=0.7)

            # Add percentage labels
            for i, (casual, registered) in enumerate(zip(weekday_totals['casual'], weekday_totals['registered'])):
                plt.text(i, casual / 2 + registered, f'{percentage_casual.iloc[i]:.1f}%', ha='center', va='center', color='white', fontsize=8)
                plt.text(i, casual + registered / 2, f'{percentage_registered.iloc[i]:.1f}%', ha='center', va='center', color='white', fontsize=8)

            plt.title('Bike Rentals by User Type and Day of the Week')
            plt.xlabel('Day of the Week')
            plt.ylabel('Total Number of Rentals')
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(plt)

        st.markdown("""
        ### Hourly 
        - Commuter patterns visible, with spikes during typical rush hours, reflecting work (or school)-related usage.""") 
        with st.expander("Chart"):
            st.header('Hourly Bike Rentals')
            st.markdown('comentario de la grafica')
            hourly_data = data.groupby('hr').agg({'casual': 'sum', 'registered': 'sum', 'cnt': 'sum'})

            # Calculate the percentage for each component
            total = hourly_data['casual'] + hourly_data['registered']
            percentage_casual = hourly_data['casual'] / total * 100
            percentage_registered = hourly_data['registered'] / total * 100

            # Create a stacked bar chart
            plt.figure(figsize=(10, 5))
            p1 = plt.bar(hourly_data.index, hourly_data['casual'], label='Casual', alpha=0.7)
            p2 = plt.bar(hourly_data.index, hourly_data['registered'], bottom=hourly_data['casual'], label='Registered', alpha=0.7)

            # Add percentage labels
            for i, (casual, registered) in enumerate(zip(hourly_data['casual'], hourly_data['registered'])):
                plt.text(i, casual / 2, f'{percentage_casual.iloc[i]:.1f}%', ha='center', va='center', color='white', fontsize=8)
                plt.text(i, casual + registered / 2, f'{percentage_registered.iloc[i]:.1f}%', ha='center', va='center', color='white', fontsize=8)

            plt.title('Hourly Distribution of Rentals (Casual vs. Registered)')
            plt.ylabel('Number of Users')
            plt.xlabel('Hour')
            plt.xticks(hourly_data.index)
            plt.legend()
            st.pyplot(plt)

    with tab3:
        st.markdown("""
            ### Outliers
        - Outliers predominantly in 2012, aligned with registered users and favorable weather conditions.
        - These data points are crucial, as they represent peak demand scenarios.""")

    with tab4:
        st.markdown("""
            ### Feature Engineering 
         - Introduction of innovative features like day parts and weekend indicators to capture user patterns.
         - Application of sine and cosine transformations to 'hour' variable for capturing cyclical nature.
         - Treatment of 'month' as a categorical variable, recognizing the impact of seasonal changes.""")
       
    with tab5:
        st.markdown("""
            ### Correlation Insights
    - 'Holiday' and 'workingday' variables showing limited predictive power.
    - Despite low correlation, 'windspeed' emerges as an influential predictor.
    - Discarding 'is_weekend' due to its negligible impact on model performance.""")
        with st.expander("Chart"):
            correlation_matrix = data.corr()
            plt.figure(figsize=(14, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
            plt.title('Correlation Matrix')
            st.pyplot(plt)

    with tab6:
        st.markdown("""
            ### Feature Importance Analysis
        - Significant impact of short-term trends captured by a 3-hour moving average.
        - The cyclical representation of time (sine and cosine transformations of 'hour') is pivotal.
        - Emphasis on real-time data for accurate forecasting, given the model's reliance on recent trends.""")





    st.header('Comprehensive Model Analysis')

# Introduction text for model analysis
    st.markdown("""
    ### Selected Variables
    The following variables were instrumental in model development:
    - **Numerical Variables**: Sine and cosine of 'hour', humidity, windspeed, and 'atemp' (temperature).
    - **Categorical Variables** (One-Hot Encoded): 'month', day part, weekday, and weather situation.
                
    ### Methodological Approach
    Utilizing PyCaret, we compared various models and conducted an in-depth analysis of five contenders: 
    CatBoost, XGBoost, LightGBM, Extra Trees, and Random Forest.""")

    st.markdown("""
    #### CatBoost Regressor:
    - **Accuracy**: Exceptional, with the lowest mean RMSE of 67.9117 and an R2 score of 0.8583.
    - **Strength**: Remarkable consistency across different data folds, ensuring generalization.""")

    st.markdown("""
    #### XGBoost Regressor:
    - **Prediction Quality**: Strong, with a mean RMSE of 69.4719 and an R2 score of 0.8517.
    - **Characteristic**: Minor overfitting tendencies but retains reliable performance.""")

    st.markdown("""
    #### LightGBM Regressor:
    - **Performance**: Competitive, with a mean RMSE of 70.9699 and an R2 score of 0.8453.
    - **Observation**: Comparable to XGBoost in terms of consistency but slightly less accurate.""")

    st.markdown("""
    #### Extra Trees Regressor:
    - **Stability**: Consistent, evidenced by a mean RMSE of 75.0255 and an R2 score of 0.8272.
    - **Suitability**: Effective for varied datasets despite marginally higher error rates.""")

    st.markdown("""
    #### Random Forest Regressor:
    - **Efficiency**: Notable, with an R2 score of 0.8358 and a mean RMSE of 73.1146.
    - **Tendency**: Potential overfitting indicated by perfect training scores, yet strong in validation.""")
    


    # Placeholder for model explanation, charts, or visuals
    # st.image('path_to_model_visualization', use_container_width=True)

# Predictions Page
elif st.session_state['current_page'] == "predictions":
    st.markdown('---') 
    # Content for the Predictions page
    st.title('Bike Demand Predictions')
    st.header('Get Real-Time Predictions of Bicycle Demand')

    # Introduction text for the Predictions page
    st.markdown(
        """
        Use this page to input variables and receive real-time predictions of bicycle demand. 
        This predictive tool can help the transportation department optimize provisioning and control costs.
        """
    )

    def load_pycaret_model():
        return load_model('final_model')

    model = load_pycaret_model()

    month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    weather_situation_names = {
    1: 'Clear, Few clouds, Partly cloudy, Partly cloudy',
    2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
    3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
    4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'}

    mnth = st.selectbox('Month', options=list(range(1, 13)), format_func=lambda x: month_names[x])
    weathersit = st.selectbox('Weather Situation', options=list(range(1, 5)), format_func=lambda x: weather_situation_names[x])
    atemp = st.slider('Feeling Temperature', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    hum = st.slider('Humidity', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    windspeed = st.slider('Windspeed', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    hr = st.slider('Hour', min_value=0, max_value=23, step=1, value=12)
    weekday_name = st.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    sin_hour = np.sin(hr * (2. * np.pi / 24))
    cos_hour = np.cos(hr * (2. * np.pi / 24))

    if hr < 6:
        day_part = 'Night'
    elif hr < 12:
        day_part = 'Morning'
    elif hr < 18:
        day_part = 'Afternoon'
    else:
        day_part = 'Evening'


    if st.button('Predict'):
    # Prepare the input data with correct column names
        input_data = pd.DataFrame([[mnth, weathersit, atemp, hum, windspeed, sin_hour, cos_hour, day_part, weekday_name]],
                                columns=['mnth', 'weathersit', 'atemp', 'hum', 'windspeed', 'hr_sin', 'hr_cos', 'day_part', 'weekday_name'])

    # Making a prediction
        prediction = model.predict(input_data)

    # Display the prediction
        st.write(f'Predicted number of bikes needed: {int(prediction[0])}')


# Footer or additional information (common to all pages)
st.markdown('---')

st.markdown(
    """
    * Built by [Consulting Consultants ](https://www.yourconsultancy.com)
    * Contact us at contact@consultingconsultants.com
    """
)
