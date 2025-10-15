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

model, label_encoders, feature_names = load_model()

# Page config
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 HDB Resale Price Predictor")
st.markdown("### Predict HDB flat prices in Singapore using Machine Learning")

# Sidebar for inputs
st.sidebar.header("Enter Flat Details")

# Get unique values for dropdowns
town = st.sidebar.selectbox(
    "Town",
    options=sorted(label_encoders['town'].classes_)
)

flat_type = st.sidebar.selectbox(
    "Flat Type",
    options=sorted(label_encoders['flat_type'].classes_)
)

flat_model = st.sidebar.selectbox(
    "Flat Model",
    options=sorted(label_encoders['flat_model'].classes_)
)

storey_range = st.sidebar.selectbox(
    "Storey Range",
    options=sorted(label_encoders['storey_range'].classes_)
)

floor_area_sqm = st.sidebar.number_input(
    "Floor Area (sqm)",
    min_value=30.0,
    max_value=200.0,
    value=90.0,
    step=1.0
)

lease_commence_date = st.sidebar.number_input(
    "Lease Commence Year",
    min_value=1960,
    max_value=2024,
    value=1990,
    step=1
)

current_year = datetime.now().year
flat_age = current_year - lease_commence_date
remaining_lease_years = 99 - flat_age

st.sidebar.markdown(f"**Calculated Flat Age:** {flat_age} years")
st.sidebar.markdown(f"**Remaining Lease:** {remaining_lease_years} years")

# Predict button
if st.sidebar.button("Predict Price", type="primary"):
    # Prepare input data
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
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Price",
            value=f"${prediction:,.0f}"
        )
    
    with col2:
        price_per_sqm = prediction / floor_area_sqm
        st.metric(
            label="Price per sqm",
            value=f"${price_per_sqm:,.0f}"
        )
    
    with col3:
        st.metric(
            label="Floor Area",
            value=f"{floor_area_sqm} sqm"
        )
    
    # Additional info
    st.success("✅ Prediction complete!")
    
    st.markdown("### 📊 Flat Summary")
    summary_df = pd.DataFrame({
        'Detail': ['Town', 'Flat Type', 'Model', 'Storey Range', 'Floor Area', 
                   'Lease Start', 'Flat Age', 'Remaining Lease'],
        'Value': [town, flat_type, flat_model, storey_range, 
                  f"{floor_area_sqm} sqm", lease_commence_date, 
                  f"{flat_age} years", f"{remaining_lease_years} years"]
    })
    st.table(summary_df)
    
    # Price breakdown
    st.markdown("### 💰 Price Insights")
    monthly_payment = prediction * 0.0045  # Approximate monthly payment
    st.info(f"""
    - **Total Price**: ${prediction:,.0f}
    - **Price per sqm**: ${price_per_sqm:,.0f}
    - **Estimated monthly loan** (2.5% interest, 25 years): ${monthly_payment:,.0f}
    """)

else:
    st.info("👈 Enter flat details in the sidebar and click 'Predict Price'")
    
    st.markdown("""
    ### About this Project
    
    This machine learning model predicts HDB resale flat prices based on:
    - **Location** (Town)
    - **Flat characteristics** (Type, Model, Size)
    - **Age and remaining lease**
    - **Storey range**
    
    **Data Source**: data.gov.sg - HDB Resale Flat Prices (Jan 2017 onwards)
    
    **Model**: XGBoost Regression with hyperparameter tuning
    
    **Features**:
    - Trained on 100,000+ real transactions
    - Optimized through RandomizedSearchCV
    - Cross-validated for reliability
    """)

# Model Performance Section
st.markdown("---")
st.markdown("### 📊 Model Performance & Evaluation")

try:
    with open('model_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Mean Absolute Error",
            value=f"${metrics['mae']:,.0f}",
            delta=f"{metrics['improvement_mae']:.1f}% improvement",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="R² Score",
            value=f"{metrics['r2']:.4f}",
            delta=f"{metrics['improvement_r2']:.1f}% improvement",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="RMSE",
            value=f"${metrics['rmse']:,.0f}",
            delta=f"{metrics['improvement_rmse']:.1f}% improvement",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="MAPE",
            value=f"{metrics['mape']:.2f}%"
        )
    
    # Expandable section for detailed evaluation
    with st.expander("📈 View Detailed Model Evaluation & Visualizations"):
        st.markdown("""
        ### Model Training Approach
        
        This model was developed using a rigorous machine learning pipeline:
        
        1. **Baseline Model**: Started with default XGBoost parameters
        2. **Hyperparameter Tuning**: Used RandomizedSearchCV with 20 iterations
        3. **Cross-Validation**: 5-fold CV to ensure generalization
        4. **Final Evaluation**: Tested on 20% holdout set
        """)
        
        # Performance comparison
        st.markdown("### 🎯 Performance Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['MAE ($)', 'RMSE ($)', 'R² Score'],
            'Baseline Model': [
                f"${metrics['baseline_mae']:,.0f}",
                f"${metrics['baseline_rmse']:,.0f}",
                f"{metrics['baseline_r2']:.4f}"
            ],
            'Optimized Model': [
                f"${metrics['mae']:,.0f}",
                f"${metrics['rmse']:,.0f}",
                f"{metrics['r2']:.4f}"
            ],
            'Improvement': [
                f"{metrics['improvement_mae']:.2f}%",
                f"{metrics['improvement_rmse']:.2f}%",
                f"{metrics['improvement_r2']:.2f}%"
            ]
        })
        st.table(comparison_df)
        
        # Cross-validation results
        st.markdown("### 🔄 Cross-Validation Results")
        st.info(f"""
        **5-Fold Cross-Validation MAE**: ${metrics['cv_mae']:,.0f} (±${metrics['cv_std']:,.0f})
        
        This shows the model performs consistently across different data splits, 
        indicating good generalization to unseen data.
        """)
        
        # Visualizations
        st.markdown("### 📊 Evaluation Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                st.image('predicted_vs_actual.png', caption='Predicted vs Actual Prices', use_container_width=True)
                st.markdown("**Interpretation**: Points close to the red line indicate accurate predictions.")
            except:
                st.warning("Visualization not found. Run train_model.py to generate plots.")
            
            try:
                st.image('error_distribution.png', caption='Error Distribution', use_container_width=True)
                st.markdown("**Interpretation**: Bell-shaped curve centered at zero indicates unbiased predictions.")
            except:
                pass
        
        with col2:
            try:
                st.image('residual_plot.png', caption='Residual Analysis', use_container_width=True)
                st.markdown("**Interpretation**: Random scatter around zero line shows good model fit.")
            except:
                st.warning("Visualization not found. Run train_model.py to generate plots.")
            
            try:
                st.image('feature_importance.png', caption='Feature Importance', use_container_width=True)
                st.markdown("**Interpretation**: Shows which features most influence price predictions.")
            except:
                pass
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        st.success(f"""
        - ✅ Model achieves **{metrics['r2']:.1%}** accuracy (R² score)
        - ✅ Average prediction error: **${metrics['mae']:,.0f}** ({metrics['mape']:.2f}% MAPE)
        - ✅ Hyperparameter tuning improved performance by **{metrics['improvement_mae']:.1f}%**
        - ✅ Cross-validation confirms model stability: **${metrics['cv_mae']:,.0f}** MAE
        - ✅ Model generalizes well to unseen data
        """)
        
        st.markdown("### 🔧 Hyperparameters Used")
        st.code("""
        Best parameters from RandomizedSearchCV:
        - Optimized across 20 parameter combinations
        - Selected based on MAE performance
        - 3-fold cross-validation during search
        """)

except FileNotFoundError:
    st.warning("⚠️ Model metrics not found. Please run `python train_model.py` first to generate evaluation metrics.")
    st.info("This will create performance visualizations and metrics that demonstrate the model's accuracy.")

# Technical Details Section
st.markdown("---")
st.markdown("### 🔬 Technical Details")

with st.expander("Click to view technical implementation details"):
    st.markdown("""
    #### Features Used
    - **Categorical**: Town, Flat Type, Flat Model, Storey Range
    - **Numerical**: Floor Area, Lease Commence Date, Flat Age, Remaining Lease Years
    - **Temporal**: Year, Month
    
    #### Data Processing
    - **Encoding**: Label Encoding for categorical variables
    - **Feature Engineering**: Derived features like flat age and remaining lease
    - **Train-Test Split**: 80-20 split with random state for reproducibility
    
    #### Model Architecture
    - **Algorithm**: XGBoost (Extreme Gradient Boosting)
    - **Type**: Regression
    - **Optimization**: RandomizedSearchCV with 20 iterations
    - **Validation**: 5-fold cross-validation
    
    #### Evaluation Metrics
    - **MAE** (Mean Absolute Error): Average prediction error in dollars
    - **RMSE** (Root Mean Squared Error): Penalizes larger errors more heavily
    - **R²** (R-squared): Proportion of variance explained by the model
    - **MAPE** (Mean Absolute Percentage Error): Average error as percentage
    
    #### Libraries Used
```python
    - pandas: Data manipulation
    - scikit-learn: Model training and evaluation
    - xgboost: Gradient boosting implementation
    - streamlit: Web interface
    - matplotlib/seaborn: Visualizations
    """)
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Python, XGBoost, and Streamlit</p>
    <p>Data Source: <a href='https://data.gov.sg' target='_blank'>data.gov.sg</a> - HDB Resale Flat Prices</p>
    <p><small>Last updated: October 2025</small></p>
</div>
""", unsafe_allow_html=True)
    