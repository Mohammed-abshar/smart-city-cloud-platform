import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from datetime import datetime, timedelta

# 1. Page Configuration
st.set_page_config(page_title="Smart City Cloud Dashboard", layout="wide")
st.title("☁️ Cloud Cost Intelligence Platform for Smart City")
st.markdown("Analyze historical cloud billing data and predict future infrastructure costs using Machine Learning.")

# 2. Smart Load Data (Auto-generates 1000 datasets if missing)
@st.cache_data
def load_data():
    file_name = 'smart_city_cloud_billing.csv'
    
    if not os.path.exists(file_name):
        np.random.seed(42)
        dates = [datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(1000)]
        services = ['Traffic Management', 'Smart Lighting', 'Waste Management', 'Public Wi-Fi', 'Security Cameras']
        items = ['Compute Engine', 'Cloud Storage', 'Database', 'Data Analytics', 'Networking']
        
        data = {
            'Date': dates,
            'Service_Type': [random.choice(services) for _ in range(1000)],
            'Item': [random.choice(items) for _ in range(1000)],
            'Cost_USD': np.random.uniform(10.0, 500.0, 1000).round(2)
        }
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# 3. Core Calculations & Prediction Logic
total_cost = df['Cost_USD'].sum()
most_expensive_idx = df['Cost_USD'].idxmax()
most_expensive_item = df.loc[most_expensive_idx]

# Linear Regression AI Prediction
df_sorted = df.sort_values('Date')
monthly_costs = df_sorted.resample('ME', on='Date')['Cost_USD'].sum().reset_index()

monthly_costs['Date_Num'] = monthly_costs['Date'].map(lambda x: x.toordinal())
next_month = monthly_costs['Date'].iloc[-1] + pd.DateOffset(months=1)
next_month_num = next_month.toordinal()

coefficients = np.polyfit(monthly_costs['Date_Num'], monthly_costs['Cost_USD'], 1)
trend_line = np.poly1d(coefficients)
predicted_cost = trend_line(next_month_num)

# 4. Dashboard Layout: Top Metric Cards
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

col1.metric("Total Historical Spend", f"${total_cost:,.2f}")
col2.metric("Next Month AI Forecast", f"${predicted_cost:,.2f}")
col3.metric("Highest Single Charge", f"${most_expensive_item['Cost_USD']:,.2f}", f"{most_expensive_item['Item']}", delta_color="inverse")

st.divider()

# 5. Dashboard Layout: Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Cost Breakdown by Service")
    service_costs = df.groupby('Service_Type')['Cost_USD'].sum()
    
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.pie(service_costs, labels=service_costs.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    st.pyplot(fig1)

with col_chart2:
    st.subheader("Monthly Cost Trend & ML Prediction")
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(monthly_costs['Date'], monthly_costs['Cost_USD'], marker='o', linestyle='-', label='Historical Cost')
    ax2.plot(monthly_costs['Date'], trend_line(monthly_costs['Date_Num']), linestyle=':', color='gray', label='Trend Line')
    ax2.plot([monthly_costs['Date'].iloc[-1], next_month], 
             [monthly_costs['Cost_USD'].iloc[-1], predicted_cost], 
             marker='x', linestyle='--', color='red', label='Next Month ML Prediction')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cost (USD)')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# Optional: Display the raw dataset at the bottom
with st.expander("View Raw Cloud Billing Data"):
    st.dataframe(df)