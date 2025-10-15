import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Load model and encoders
@st.cache_resource
def load_model():
    with open('hdb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, label_encoders, feature_names

# Load training data for validation
@st.cache_data
def load_validation_data():
    """Load training data to validate combinations"""
    try:
        df = pd.read_csv('resale-flat-prices-cleaned.csv')
        return df
    except:
        st.error("❌ Cannot find resale-flat-prices-cleaned.csv. Run clean_data.py first!")
        return None

model, label_encoders, feature_names = load_model()
training_data = load_validation_data()

# Page config
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 HDB Resale Price Predictor (With Validation)")
st.markdown("### Predict HDB flat prices using validated inputs")

# Show data status
if training_data is not None:
    st.success(f"✅ Loaded {len(training_data):,} validated HDB transactions")
else:
    st.error("⚠️ Training data not loaded. Validation disabled.")

# Sidebar
st.sidebar.header("Enter Flat Details")

# STEP 1: Select Town
town = st.sidebar.selectbox(
    "🏘️ Town",
    options=sorted(label_encoders['town'].classes_),
    help="Select the HDB town"
)

# STEP 2: Filter Flat Types based on Town
if training_data is not None:
    # Only show flat types that exist in this town
    available_flat_types = training_data[training_data['town'] == town]['flat_type'].unique()
    available_flat_types = sorted([ft for ft in available_flat_types if ft in label_encoders['flat_type'].classes_])
    
    if len(available_flat_types) == 0:
        st.sidebar.error(f"❌ No data for {town}")
        st.stop()
else:
    available_flat_types = sorted(label_encoders['flat_type'].classes_)

flat_type = st.sidebar.selectbox(
    "🏠 Flat Type",
    options=available_flat_types,
    help=f"{len(available_flat_types)} flat types available in {town}"
)

# STEP 3: Filter Flat Models based on Town + Flat Type
if training_data is not None:
    available_flat_models = training_data[
        (training_data['town'] == town) & 
        (training_data['flat_type'] == flat_type)
    ]['flat_model'].unique()
    available_flat_models = sorted([fm for fm in available_flat_models if fm in label_encoders['flat_model'].classes_])
    
    if len(available_flat_models) == 0:
        st.sidebar.error(f"❌ No {flat_type} flats in {town}")
        st.stop()
else:
    available_flat_models = sorted(label_encoders['flat_model'].classes_)

flat_model = st.sidebar.selectbox(
    "🏗️ Flat Model",
    options=available_flat_models,
    help=f"{len(available_flat_models)} models available"
)

# STEP 4: Filter Storey Ranges based on Town
if training_data is not None:
    available_storey_ranges = training_data[training_data['town'] == town]['storey_range'].unique()
    available_storey_ranges = sorted([sr for sr in available_storey_ranges if sr in label_encoders['storey_range'].classes_])
    
    if len(available_storey_ranges) == 0:
        st.sidebar.error(f"❌ No storey data for {town}")
        st.stop()
else:
    available_storey_ranges = sorted(label_encoders['storey_range'].classes_)

storey_range = st.sidebar.selectbox(
    "🏢 Storey Range",
    options=available_storey_ranges,
    help=f"{len(available_storey_ranges)} storey ranges available in {town}"
)

# STEP 5: Set Floor Area Range based on actual data
if training_data is not None:
    similar_flats = training_data[
        (training_data['town'] == town) & 
        (training_data['flat_type'] == flat_type)
    ]
    
    if len(similar_flats) > 0:
        min_area = float(similar_flats['floor_area_sqm'].min())
        max_area = float(similar_flats['floor_area_sqm'].max())
        typical_area = float(similar_flats['floor_area_sqm'].median())
    else:
        min_area, max_area, typical_area = 30.0, 200.0, 90.0
else:
    min_area, max_area, typical_area = 30.0, 200.0, 90.0

floor_area_sqm = st.sidebar.number_input(
    "📐 Floor Area (sqm)",
    min_value=min_area,
    max_value=max_area,
    value=typical_area,
    step=1.0,
    help=f"Range for {flat_type} in {town}: {min_area:.0f}-{max_area:.0f} sqm"
)

# STEP 6: Set Lease Date Range based on actual data
if training_data is not None and len(similar_flats) > 0:
    min_lease = int(similar_flats['lease_commence_date'].min())
    max_lease = int(similar_flats['lease_commence_date'].max())
    typical_lease = int(similar_flats['lease_commence_date'].median())
else:
    min_lease, max_lease, typical_lease = 1960, 2024, 1990

lease_commence_date = st.sidebar.number_input(
    "📅 Lease Commence Year",
    min_value=min_lease,
    max_value=max_lease,
    value=typical_lease,
    step=1,
    help=f"Range for this area: {min_lease}-{max_lease}"
)

current_year = datetime.now().year
flat_age = current_year - lease_commence_date
remaining_lease_years = 99 - flat_age

st.sidebar.markdown(f"**🕐 Calculated Flat Age:** {flat_age} years")
st.sidebar.markdown(f"**📋 Remaining Lease:** {remaining_lease_years} years")

# VALIDATION CHECKS
st.sidebar.markdown("---")
st.sidebar.markdown("### ✅ Validation Status")

warnings = []
confidence = "High"

if training_data is not None:
    # Check if exact combination exists
    exact_match = training_data[
        (training_data['town'] == town) &
        (training_data['flat_type'] == flat_type) &
        (training_data['flat_model'] == flat_model) &
        (training_data['storey_range'] == storey_range)
    ]
    
    num_matches = len(exact_match)
    
    if num_matches == 0:
        warnings.append("⚠️ This exact combination doesn't exist in our data")
        confidence = "Low"
        st.sidebar.error(f"❌ 0 exact matches found")
    elif num_matches < 5:
        warnings.append(f"⚠️ Rare combination: Only {num_matches} similar properties")
        confidence = "Medium"
        st.sidebar.warning(f"⚠️ {num_matches} exact matches")
    else:
        st.sidebar.success(f"✅ {num_matches} exact matches")
    
    # Check floor area
    if floor_area_sqm < min_area * 0.95 or floor_area_sqm > max_area * 1.05:
        warnings.append(f"⚠️ Floor area unusual for {flat_type} in {town}")
    
    # Check lease date
    if lease_commence_date < min_lease or lease_commence_date > max_lease:
        warnings.append(f"⚠️ Lease date outside typical range")

# Display validation summary
if len(warnings) == 0:
    st.sidebar.success("✅ All inputs validated!")
else:
    st.sidebar.warning(f"⚠️ {len(warnings)} warning(s)")

# Predict button
if st.sidebar.button("🎯 Predict Price", type="primary", use_container_width=True):
    
    # Prepare input data
    try:
        input_data = pd.DataFrame({
            'town': [label_encoders['town'].transform([town])[0]],
            'flat_type': [label_encoders['flat_type'].transform([flat_type])[0]],
            'flat_model': [label_encoders['flat_model'].transform([flat_model])[0]],
            'floor_area_sqm': [floor_area_sqm],
            'lease_commence_date': [lease_commence_date],
            'year': [current_year],
            'month_num': [datetime.now().month],
            'storey_range': [label_encoders['storey_range'].transform([storey_range])[0]],
            'remaining_lease_years': [remaining_lease_years],
            'flat_age': [flat_age]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results with confidence indicator
        if confidence == "Low":
            st.error("⚠️ **LOW CONFIDENCE PREDICTION**")
            st.warning("This combination is very rare or doesn't exist in our training data. The prediction may be unreliable.")
        elif confidence == "Medium":
            st.warning("⚠️ **MEDIUM CONFIDENCE PREDICTION**")
            st.info("This combination is rare. Use this prediction with caution.")
        else:
            st.success("✅ **HIGH CONFIDENCE PREDICTION**")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="💰 Predicted Price",
                value=f"${prediction:,.0f}"
            )
        
        with col2:
            price_per_sqm = prediction / floor_area_sqm
            st.metric(
                label="📊 Price per sqm",
                value=f"${price_per_sqm:,.0f}"
            )
        
        with col3:
            st.metric(
                label="🎯 Confidence",
                value=confidence,
                delta="Based on data availability"
            )
        
        # Show warnings
        if warnings:
            st.markdown("### ⚠️ Warnings")
            for warning in warnings:
                st.warning(warning)
        
        # Flat summary
        st.markdown("### 📋 Flat Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            - **Town:** {town}
            - **Flat Type:** {flat_type}
            - **Model:** {flat_model}
            - **Storey Range:** {storey_range}
            """)
        
        with col2:
            st.markdown(f"""
            - **Floor Area:** {floor_area_sqm} sqm
            - **Lease Start:** {lease_commence_date}
            - **Flat Age:** {flat_age} years
            - **Remaining Lease:** {remaining_lease_years} years
            """)
        
        # Show similar properties if available
        if training_data is not None and len(exact_match) > 0:
            st.markdown("### 🔍 Similar Properties in Database")
            
            col1, col2, col3, col4 = st.columns(4)
            
            prices = exact_match['resale_price']
            with col1:
                st.metric("📉 Min Price", f"${prices.min():,.0f}")
            with col2:
                st.metric("📊 Median Price", f"${prices.median():,.0f}")
            with col3:
                st.metric("📈 Max Price", f"${prices.max():,.0f}")
            with col4:
                st.metric("📊 Avg Price", f"${prices.mean():,.0f}")
            
            # Show if prediction is within range
            if prices.min() <= prediction <= prices.max():
                st.success(f"✅ Your prediction (${prediction:,.0f}) falls within the historical range!")
            else:
                st.warning(f"⚠️ Your prediction is outside the historical range for similar properties.")
        
        # Price breakdown
        st.markdown("### 💰 Financial Breakdown")
        monthly_payment = prediction * 0.0045
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Total Price:** ${prediction:,.0f}
            
            **Price per sqm:** ${price_per_sqm:,.0f}
            
            **Estimated monthly loan** (2.5% interest, 25 years): 
            ${monthly_payment:,.0f}/month
            """)
        
        with col2:
            if training_data is not None and len(similar_flats) > 0:
                avg_price_similar = similar_flats['resale_price'].median()
                diff = prediction - avg_price_similar
                diff_pct = (diff / avg_price_similar) * 100
                
                if abs(diff_pct) < 5:
                    st.success(f"📊 **Fair Market Value**\n\nThis price is within 5% of similar properties (${avg_price_similar:,.0f})")
                elif diff > 0:
                    st.warning(f"📈 **Above Average**\n\nThis is {diff_pct:.1f}% higher than similar properties (${avg_price_similar:,.0f})")
                else:
                    st.info(f"📉 **Below Average**\n\nThis is {abs(diff_pct):.1f}% lower than similar properties (${avg_price_similar:,.0f})")
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("This might happen if the combination is invalid. Please check your inputs.")

else:
    # Welcome message
    st.info("👈 **Enter flat details in the sidebar and click 'Predict Price'**")
    
    st.markdown("""
    ### 🎯 About This Tool
    
    This machine learning model predicts HDB resale flat prices with **built-in validation** to ensure predictions are reliable.
    
    #### ✅ Key Features:
    - **Smart Input Filtering**: Only shows options that exist in real data
    - **Validation Warnings**: Alerts you when combinations are rare or unusual
    - **Confidence Levels**: High/Medium/Low based on data availability
    - **Similar Properties**: Shows actual prices from database for comparison
    
    #### 🛡️ Protection Against Invalid Predictions:
    - ❌ No more "TERRACE" type flats (not HDB!)
    - ❌ No more 50-story HDB buildings
    - ❌ No more impossible town-type combinations
    - ✅ Only predictions based on real data
    
    #### 📊 Data Source:
    - **Source**: data.gov.sg (Singapore Government)
    - **Dataset**: HDB Resale Flat Prices (2017-2025)
    - **Records**: 100,000+ validated transactions
    - **Model**: XGBoost with hyperparameter tuning
    """)

# Model metrics section
st.markdown("---")
st.markdown("### 📊 Model Performance")

try:
    with open('model_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Absolute Error", f"${metrics['mae']:,.0f}")
    with col2:
        st.metric("R² Score", f"{metrics['r2']:.4f}")
    with col3:
        st.metric("RMSE", f"${metrics['rmse']:,.0f}")
    with col4:
        st.metric("MAPE", f"{metrics['mape']:.2f}%")

except:
    st.warning("Model metrics not found. Run train_model_updated.py to generate.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏠 Built with ❤️ using Python, XGBoost, and Streamlit</p>
    <p>📊 Data: <a href='https://data.gov.sg' target='_blank'>data.gov.sg</a> | 🔒 All predictions validated against real data</p>
</div>
""", unsafe_allow_html=True)