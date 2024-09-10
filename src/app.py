import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
import pandas as pd
import psycopg2

# Sample function to load data (assuming you're loading data from a CSV)
@st.cache_data
def load_data():
    # Load your dataset here (CSV, PostgreSQL query, etc.)
    data = pd.read_csv("C:/Users/dell/TellCo_Investment_Insights/telecom_data.csv")
    return data

# Function to handle missing values in the dataset
def handle_missing_values(data):
    # Fill missing values with the mean or mode as appropriate
    data['Avg RTT DL (ms)'].fillna(data['Avg RTT DL (ms)'].mean(), inplace=True)
    data['Avg RTT UL (ms)'].fillna(data['Avg RTT UL (ms)'].mean(), inplace=True)
    data['Avg Bearer TP DL (kbps)'].fillna(data['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
    data['Avg Bearer TP UL (kbps)'].fillna(data['Avg Bearer TP UL (kbps)'].mean(), inplace=True)
    data['TCP DL Retrans. Vol (Bytes)'].fillna(data['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
    data['TCP UL Retrans. Vol (Bytes)'].fillna(data['TCP UL Retrans. Vol (Bytes)'].mean(), inplace=True)
    data['Handset Type'].fillna(data['Handset Type'].mode()[0], inplace=True)
    return data

# Main dashboard layout
def main():
    st.title("Telecommunication User Experience Dashboard")

    # Load the data
    data = load_data()

    # Handle missing values
    data = handle_missing_values(data)

    # Display a sample of the data
    st.write("### Sample Data")
    st.dataframe(data.head(10))

    # Sidebar filters for user interaction
    st.sidebar.header("Filters")
    location_filter = st.sidebar.multiselect(
        "Select Last Location Name",
        data['Last Location Name'].unique()
    )
    
    handset_filter = st.sidebar.multiselect(
        "Select Handset Type",
        data['Handset Type'].unique()
    )

    # Filter data based on user selections
    if location_filter:
        data = data[data['Last Location Name'].isin(location_filter)]
    if handset_filter:
        data = data[data['Handset Type'].isin(handset_filter)]

    # Display filtered data and metrics
    st.write("### Filtered Data")
    st.dataframe(data)

    # Display basic statistics
    st.write("### Summary Statistics")
    st.write(data.describe())

    # Visualization examples
    st.write("### Visualizations")

    st.bar_chart(data['Avg Bearer TP DL (kbps)'].value_counts())

    st.line_chart(data[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']])

# Run the app
if __name__ == "__main__":
    main()

# import numpy as np
# from sklearn.metrics import euclidean_distances
# from sklearn.cluster import KMeans

# # Load the clustering data (from Task 3.4)
# X = data[['tcp_retransmission', 'rtt', 'throughput']]

# # Apply KMeans clustering (from previous task)
# kmeans = KMeans(n_clusters=3, random_state=42)
# data['cluster'] = kmeans.fit_predict(X)

# # Get the centroids of each cluster
# centroids = kmeans.cluster_centers_

# # Assign the "least engaged" and "worst experience" cluster as the cluster with the maximum values
# least_engaged_cluster = np.argmax(centroids[:, 2])  # Assuming throughput is the engagement metric
# worst_experience_cluster = np.argmax(centroids[:, 0])  # Assuming tcp_retransmission is experience metric

# # Calculate the Euclidean distance (engagement score and experience score)
# data['engagement_score'] = euclidean_distances(X, [centroids[least_engaged_cluster]]).flatten()
# data['experience_score'] = euclidean_distances(X, [centroids[worst_experience_cluster]]).flatten()

# st.write(data[['handset_type', 'engagement_score', 'experience_score']])

# # Calculate satisfaction score as the average of engagement and experience scores
# data['satisfaction_score'] = (data['engagement_score'] + data['experience_score']) / 2

# # Sort by satisfaction score and report the top 10 satisfied customers
# top_10_satisfied = data.sort_values(by='satisfaction_score', ascending=True).head(10)
# st.write("Top 10 Satisfied Customers")
# st.write(top_10_satisfied[['handset_type', 'satisfaction_score']])

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Select features (tcp_retransmission, rtt, throughput) and target (satisfaction_score)
# X_features = data[['tcp_retransmission', 'rtt', 'throughput']]
# y_target = data['satisfaction_score']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=42)

# # Create a linear regression model
# regression_model = LinearRegression()
# regression_model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = regression_model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# st.write(f"Mean Squared Error of the Regression Model: {mse}")

# # Display coefficients
# coefficients = pd.DataFrame(regression_model.coef_, X_features.columns, columns=['Coefficient'])
# st.write("Regression Coefficients")
# st.write(coefficients)

# # Aggregate average satisfaction and experience scores per cluster
# cluster_aggregates = data.groupby('score_cluster')[['satisfaction_score', 'experience_score']].mean()
# st.write("Average Satisfaction & Experience Score per Cluster")
# st.write(cluster_aggregates)

# from db_conn import create_table
# # Create table for storing user engagement, experience, and satisfaction
# def create_table(conn):
#     create_table_query = """
#     CREATE TABLE IF NOT EXISTS user_satisfaction (
#         user_id SERIAL PRIMARY KEY,
#         user_name VARCHAR(255),
#         engagement_score FLOAT,
#         experience_score FLOAT,
#         satisfaction_score FLOAT
#     );
#     """
#     cursor = conn.cursor()
#     cursor.execute(create_table_query)
#     conn.commit()
#     cursor.close()

# # Insert data into the table
# def insert_data(conn, user_data):
#     cursor = conn.cursor()
#     insert_query = """
#     INSERT INTO user_satisfaction (user_name, engagement_score, experience_score, satisfaction_score)
#     VALUES (%s, %s, %s, %s);
#     """
#     for row in user_data.itertuples(index=False):
#         cursor.execute(insert_query, (row.user_name, row.engagement_score, row.experience_score, row.satisfaction_score))
#     conn.commit()
#     cursor.close()
# # Create connection to the PostgreSQL database
# conn = create_connection()

# # Create table
# create_table(conn)

# # Insert data into the table
# insert_data(conn, user_data)

# # Close connection
# conn.close()
# print("Data exported to PostgreSQL successfully!")