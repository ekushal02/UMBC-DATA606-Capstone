# Forecasting Household Energy Consumption Using Time Series and Machine Learning

**Prepared for**  
UMBC Data Science Master Degree Capstone  
Dr. Chaojie (Jay) Wang  

**Author**  
Kushal Erramilli

**GitHub Repository**  
https://github.com/ekushal02/UMBC-DATA606-Capstone/tree/main

**LinkedIn Profile**  
https://www.linkedin.com/in/kushalerramilli/ 
 

---

## 1. Background

### What is this project about?

Accurate energy consumption forecasting is essential for modern energy management systems, smart grids, and sustainability efforts. With the widespread use of smart meters and high-frequency monitoring devices, electricity usage data is now available at very short time intervals. This enables a detailed examination of consumption patterns at the household level.

This project focuses on **forecasting short-term household electricity consumption** by combining **time series analysis** with **time-aware machine learning techniques**. The study uses historical electricity usage data collected at **one-minute intervals over several years** to identify patterns, seasonal trends, and nonlinear relationships that influence household energy demand.

Rather than attempting to generalize results across different populations, this study is designed as a **household-level case study and proof of concept**, demonstrating how data-driven forecasting models can support improved energy efficiency and informed demand management decisions.

---

### Why does it matter?

Household electricity consumption accounts for a substantial portion of overall energy demand and directly contributes to peak load pressure on electrical grids. Inaccurate demand forecasting can result in energy waste, increased operational costs, and higher carbon emissions.

Short-term energy forecasting at the household level enables:

- Identification of peak consumption periods  
- Improved energy efficiency and load shifting  
- Support for demand-response strategies  
- Data-driven insights for smart energy management systems  

Although this project focuses on a single household, the **analytical framework and modeling techniques are scalable** and can be applied to larger smart-meter datasets, making the findings relevant for future urban and utility-scale energy analysis.

---

### Research Questions

1. Can short-term household electricity consumption be accurately forecasted using historical time series data?  
2. What daily, weekly, and seasonal patterns are present in household energy usage?  
3. Which sub-metered household systems contribute most to overall energy consumption and variability?  
4. Do time-aware machine learning models outperform simple baseline forecasting approaches for short-term prediction?  

---

## 2. Data

### Data Source

The dataset used in this project is the **Individual Household Electric Power Consumption** dataset obtained from the **UCI Machine Learning Repository**.

The dataset contains detailed measurements of electric power consumption collected from a **single household located in Sceaux, France**, recorded at a **one-minute sampling rate** over a period of nearly **four years**.

---

### Data Size and Structure

- **File name:** `household_power_consumption.csv`  
- **File size:** approximately **126.8 MB**  
- **Number of observations:** **2,075,259 rows**  
- **Number of variables:** **9 columns**  
- **Time period covered:** **December 2006 to November 2010 (47 months)**  

---

### Unit of Observation

Each row represents **one minute of electricity usage measurements** for a single household.

---

### Dataset Characteristics

- Multivariate **time-series** dataset  
- All variables are **continuous numerical features**  
- Approximately **1.25% missing values**  
- **No missing timestamps**  

---

### Data Dictionary

| Column Name | Data Type | Description | Units / Values |
|------------|----------|-------------|----------------|
| Date | Date | Date of observation | dd/mm/yyyy |
| Time | Time | Time of observation | hh:mm:ss |
| Global_active_power | Continuous | Household global minute-averaged active power | kilowatt |
| Global_reactive_power | Continuous | Household global minute-averaged reactive power | kilowatt |
| Voltage | Continuous | Minute-averaged voltage | volt |
| Global_intensity | Continuous | Household global minute-averaged current intensity | ampere |
| Sub_metering_1 | Continuous | Kitchen appliance energy consumption | watt-hour |
| Sub_metering_2 | Continuous | Laundry room appliance energy consumption | watt-hour |
| Sub_metering_3 | Continuous | Water heater and air-conditioner energy consumption | watt-hour |

---

### Target Variable

- **Target / Label:** `Global_active_power`

This variable represents the **total active power consumption** of the household and serves as the **primary outcome variable** for forecasting.

---

### Feature Variables

The following variables will be considered as predictors in the forecasting models:

- Global_reactive_power  
- Voltage  
- Global_intensity  
- Sub_metering_1  
- Sub_metering_2  
- Sub_metering_3  
- Time-derived features *(hour of day, day of week, weekend indicator)*  
- Lagged and rolling features derived from historical energy consumption  

These features allow the models to capture both **instantaneous power usage** and **temporal dependencies** inherent in household energy consumption patterns.

---

