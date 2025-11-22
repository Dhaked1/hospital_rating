import streamlit as st
import pandas as pd

def load_data():
    df = pd.read_csv("Hospital General Information.csv", encoding='ISO-8859-1')
    return df

df = load_data()
df = df[df['Hospital Type'].notna()]
df = df[df['Mortality national comparison'].notna()]
df = df[df['Emergency Services'].notna()]
df = df[df['Hospital Name'].notna()]
df = df[df['Safety of care national comparison'].notna()]

# Convert 'Hospital overall rating' to numeric
df['Hospital overall rating'] = pd.to_numeric(df['Hospital overall rating'], errors='coerce')

st.title("Hospital Recommendation System")

# UI components
hospital_types = df['Hospital Type'].dropna().unique()
selected_type = st.selectbox("Select Hospital Type", sorted(hospital_types))
saftey_types = df['Safety of care national comparison'].dropna().unique()
saftey_type = st.selectbox("Safety of care national comparison",sorted(saftey_types))
patient_types = df['Patient experience national comparison'].dropna().unique()
patient_type = st.selectbox("Patient experience national comparison",sorted(patient_types))
state_types = df['State'].dropna().unique()
state_type = st.selectbox("State",sorted(state_types))
effective_types = df['Effectiveness of care national comparison'].dropna().unique()
effective_type = st.selectbox('Effectiveness of care national comparison',sorted(effective_types))
city_types = df['City'].dropna().unique()
city_type = st.selectbox("City",sorted(city_types))
hospital_name = df['Hospital Name'].dropna().unique()
hospital_name_type = st.selectbox("Hospital Name",sorted(hospital_name))


mortality_type = st.selectbox(
    "Select Mortality national comparison",
    ["All"] + sorted(df['Mortality national comparison'].dropna().unique().tolist())
)

emergency_filter = st.radio("Are you want Emergency Services?", ["Yes", "No"])

rating_filter = st.slider("Select Minimum Rating", 0, 5, 0)

# Filtering
filtered_df = df[df['Hospital Type'] == selected_type]
filtered_df = df[df['Safety of care national comparison'] == saftey_type]
filtered_df = df[df['Patient experience national comparison'] == patient_type]
filtered_df = df[df['State'] == state_type]
filtered_df = df[df['City'] == city_type]
filtered_df = df[df['Hospital Name'] == hospital_name_type]

filtered_df = df[df['Effectiveness of care national comparison'] == effective_type]


if mortality_type != "All":
    filtered_df = filtered_df[filtered_df['Mortality national comparison'] == mortality_type]

filtered_df = filtered_df[filtered_df['Emergency Services'] == emergency_filter]
filtered_df = filtered_df[filtered_df['Hospital overall rating'] >= rating_filter]

# Display results
st.subheader("Matching Hospitals:")
st.write(f"Found {len(filtered_df)} hospitals")

st.dataframe(filtered_df[[
    'Provider ID', 'Hospital Name', 'Address', 'City', 'State',
    'ZIP Code', 'County Name', 'Phone Number', 'Emergency Services'
]])

