# Health Facility Resilience to Power Outages in the United States

Developed a two-part machine learning model that forecasts electricity demand and detects power grid fault risk for U.S. health facilities during natural disasters. Applied regression, classification, and real-time weather APIs as part of AI4ALL’s Ignite accelerator on health system resilience.

<div>
    <a href="https://www.loom.com/share/54374adc48184bd3bb7ed054d1a96d50">
      <p>Video Walkthrough</p>
    </a>
    <a href="https://www.loom.com/share/54374adc48184bd3bb7ed054d1a96d50">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/54374adc48184bd3bb7ed054d1a96d50-b48bf543163c04c2-full-play.gif">
    </a>
  </div>
  
## Problem Statement 

Natural disasters have historically posed a significant threat to the U.S. electricity supply, frequently causing widespread power outages. Severe weather is the primary reason behind most large-scale blackouts, with states like Florida, California, New York, and Michigan experiencing millions of affected customers. The risk to the grid is further compounded by the potential for electromagnetic events and cyber-physical attacks, which have historically contributed to a substantial portion of outages and could prolong disruptions, especially if they coincide with a natural disaster. By leveraging machine learning, energy and weather data, this project aims to strengthen the resilience of healthcare infrastructure during emergencies, ensuring uninterrupted care.

## Key Results 

1. Developed a dual-model pipeline for proactive power failure mitigation
   - Integrated a weather-based electricity consumption prediction model using linear regression
   - Built a Random Forest classifier to detect potential grid fault events based on electrical features
2. Connected weather forecasting with fault detection
   - For each of the next 16 days, the system simulates energy demand and grid conditions to predict fault likelihoods
3. Enabled synthetic data generation and scenario testing
   - Created realistic voltage, current, power, and frequency data to stress-test the fault prediction model

## Methodologies 

To develop a robust prediction system for health facility power resilience, we combined multiple data sources and machine learning techniques:

- ***Data Collection and Processing***:
Integrated real-time weather data via the Open-Meteo API, geographic coordinates from Nominatim, and historical energy consumption datasets to capture environmental and operational factors impacting power usage.

- ***Energy Consumption Prediction Model***:
Trained a Linear Regression model with standardized features including temperature, humidity, wind speed, and average past consumption to forecast daily electricity usage for the next 16 days.

- ***Fault Event Classification***:
Built a Random Forest classifier trained on grid asset data, incorporating electrical measurements (voltage, current, power, frequency) and load types, to predict the likelihood of fault events.

- ***Synthetic Data Generation***:
Created synthetic sensor data based on statistical properties of training data to complement predicted consumption, enabling realistic scenario analysis for fault prediction.

- ***Model Integration Pipeline***:
Developed a cohesive pipeline that takes user input for location and past consumption, fetches weather forecasts, predicts daily consumption, simulates sensor data, and outputs fault event predictions per day.

- ***Evaluation and Validation***:
Employed standard metrics such as R² and RMSE for the regression model, and accuracy, precision, recall, and F1-score for the classifier to validate model performance.

## Data Sources 

- [Weather and Renewable Energy Data](https://www.kaggle.com/code/samanemami/weather-and-renewable-energy-analysis/input)  
- [Smart Grid Asset Monitoring Dataset](https://www.kaggle.com/datasets/ziya07/smart-grid-asset-monitoring-dataset)
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs)

## Technologies Used 

- Python  
- pandas  
- NumPy  
- scikit-learn
- joblib  
- pickle  
- requests  
- requests-cache  
- retry-requests  
- openmeteo-requests  
- Streamlit (for frontend user interface) 

## Authors 

*This project was completed in collaboration with:*
- *David Ningtang ([djnintang@gmail.com](mailto:djnintang@gmail.com))*
- *Chelsea Ross ([Chelsearosscr21@gmail.com](mailto:Chelsearosscr21@gmail.com))*
- *Sumodha Pokhrel ([sp230@rice.edu](mailto:sp230@rice.edu))*
- *Gulzira Abudula ([gzabudula21@gmail.com](gzabudula21@gmail.com))*
