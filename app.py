import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🐝 Honeybee Colony Health Predictor",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #FFE135, #FFAB00);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f8ff;
        padding: 1rem;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for data
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.rf_model = None
    st.session_state.lr_model = None
    st.session_state.scaler = None
    st.session_state.data = None

def create_synthetic_data():
    """Create the same synthetic dataset as in the original project"""
    np.random.seed(42)
    n_samples = 300
    
    colony_size = np.random.normal(15000, 3000, n_samples)
    colony_size = np.clip(colony_size, 5000, 25000)
    
    temperature = np.random.normal(22, 5, n_samples)
    temperature = np.clip(temperature, 10, 35)
    
    rainfall = np.random.normal(50, 20, n_samples)
    rainfall = np.clip(rainfall, 10, 120)
    
    queen_age = np.random.randint(1, 4, n_samples)
    
    honey_production = (
        colony_size * 0.002 +
        temperature * 1.2 +
        rainfall * 0.1 +
        (4 - queen_age) * 5 +
        np.random.normal(0, 5, n_samples)
    )
    
    honey_production = np.clip(honey_production, 5, 80)
    
    return pd.DataFrame({
        'colony_size': colony_size,
        'temperature': temperature,
        'rainfall': rainfall,
        'queen_age': queen_age,
        'honey_production': honey_production
    })

def train_models():
    """Train both Random Forest and Linear Regression models"""
    if st.session_state.model_trained:
        return
    
    # Create dataset
    data = create_synthetic_data()
    st.session_state.data = data
    
    # Prepare features and target
    X = data[['colony_size', 'temperature', 'rainfall', 'queen_age']]
    y = data['honey_production']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Store in session state
    st.session_state.rf_model = rf_model
    st.session_state.lr_model = lr_model
    st.session_state.scaler = scaler
    st.session_state.model_trained = True
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.X_test_scaled = X_test_scaled

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🐝 Honeybee Colony Health Predictor 🍯</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict honey production based on colony characteristics using Machine Learning</p>', unsafe_allow_html=True)
    
    # Train models on first load
    with st.spinner('🔄 Training ML models...'):
        train_models()
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">🎛️ Colony Parameters</h2>', unsafe_allow_html=True)
    
    # Input controls with realistic ranges
    colony_size = st.sidebar.slider(
        "🐝 Colony Size (Number of Bees)",
        min_value=5000,
        max_value=25000,
        value=15000,
        step=500,
        help="Typical range: 5,000 - 25,000 bees"
    )
    
    temperature = st.sidebar.slider(
        "🌡️ Average Temperature (°C)",
        min_value=10.0,
        max_value=35.0,
        value=22.0,
        step=0.5,
        help="Optimal range for honey production: 20-25°C"
    )
    
    rainfall = st.sidebar.slider(
        "🌧️ Monthly Rainfall (mm)",
        min_value=10.0,
        max_value=120.0,
        value=50.0,
        step=5.0,
        help="Moderate rainfall supports nectar flow"
    )
    
    queen_age = st.sidebar.selectbox(
        "👑 Queen Age (Years)",
        options=[1, 2, 3],
        index=0,
        help="Younger queens typically lead to more productive colonies"
    )
    
    # Model selection
    model_choice = st.sidebar.radio(
        "🤖 Choose Model",
        ["Random Forest", "Linear Regression"],
        help="Random Forest typically performs better for this dataset"
    )
    
    # Predict button
    predict_button = st.sidebar.button(
        "🔮 Predict Honey Production",
        type="primary",
        use_container_width=True
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">📊 Input Summary</h3>', unsafe_allow_html=True)
        
        # Display inputs in a nice format
        input_data = {
            "Parameter": ["Colony Size", "Temperature", "Rainfall", "Queen Age"],
            "Value": [f"{colony_size:,} bees", f"{temperature}°C", f"{rainfall} mm", f"{queen_age} years"],
            "Status": ["🟢 Optimal" if 12000 <= colony_size <= 20000 else "🟡 Acceptable",
                      "🟢 Optimal" if 20 <= temperature <= 25 else "🟡 Acceptable",
                      "🟢 Optimal" if 40 <= rainfall <= 70 else "🟡 Acceptable",
                      "🟢 Optimal" if queen_age == 1 else "🟡 Acceptable"]
        }
        
        input_df = pd.DataFrame(input_data)
        st.table(input_df)
        
        # Show feature importance
        if st.session_state.model_trained and model_choice == "Random Forest":
            st.markdown('<h3 class="sub-header">🎯 Feature Importance</h3>', unsafe_allow_html=True)
            
            feature_names = ['Colony Size', 'Temperature', 'Rainfall', 'Queen Age']
            importance = st.session_state.rf_model.feature_importances_
            
            fig_importance = px.bar(
                x=importance,
                y=feature_names,
                orientation='h',
                title='Feature Importance in Honey Production',
                color=importance,
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        if predict_button and st.session_state.model_trained:
            # Prepare input for prediction
            input_array = np.array([[colony_size, temperature, rainfall, queen_age]])
            
            # Make prediction
            if model_choice == "Random Forest":
                prediction = st.session_state.rf_model.predict(input_array)[0]
            else:
                input_scaled = st.session_state.scaler.transform(input_array)
                prediction = st.session_state.lr_model.predict(input_scaled)[0]
            
            # Display prediction
            st.markdown(
                f'''
                <div class="prediction-box">
                    <h2>🍯 Predicted Honey Production</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:.1f} kg</h1>
                    <p>per month using {model_choice}</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
            
            # Prediction insights
            st.markdown('<h3 class="sub-header">💡 Insights</h3>', unsafe_allow_html=True)
            
            if prediction > 50:
                st.success("🎉 Excellent production expected! Your colony parameters are well-optimized.")
            elif prediction > 35:
                st.info("👍 Good production expected. Consider optimizing temperature and colony size.")
            else:
                st.warning("⚠️ Below average production. Consider improving colony conditions.")
            
            # Show confidence metrics
            if model_choice == "Random Forest":
                r2 = r2_score(st.session_state.y_test, 
                             st.session_state.rf_model.predict(st.session_state.X_test))
                mae = mean_absolute_error(st.session_state.y_test, 
                                        st.session_state.rf_model.predict(st.session_state.X_test))
            else:
                r2 = r2_score(st.session_state.y_test, 
                             st.session_state.lr_model.predict(st.session_state.X_test_scaled))
                mae = mean_absolute_error(st.session_state.y_test, 
                                        st.session_state.lr_model.predict(st.session_state.X_test_scaled))
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Model Accuracy (R²)", f"{r2:.3f}", help="Higher is better (max 1.0)")
            with col_metric2:
                st.metric("Average Error", f"{mae:.1f} kg", help="Lower is better")
        
        else:
            st.markdown(
                '''
                <div class="info-box">
                    <h3>🚀 How to Use</h3>
                    <ol>
                        <li>Adjust the colony parameters in the sidebar</li>
                        <li>Choose your preferred ML model</li>
                        <li>Click "Predict Honey Production"</li>
                        <li>View your prediction and insights!</li>
                    </ol>
                </div>
                ''',
                unsafe_allow_html=True
            )
    
    # Additional visualizations
    if st.session_state.model_trained:
        st.markdown('<h2 class="sub-header">📈 Dataset Insights</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔍 Data Distribution", "🔗 Correlations", "📊 Model Comparison"])
        
        with tab1:
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                fig_hist = px.histogram(
                    st.session_state.data, 
                    x='honey_production', 
                    nbins=20, 
                    title='Distribution of Honey Production',
                    color_discrete_sequence=['#FFB000']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_viz2:
                fig_scatter = px.scatter(
                    st.session_state.data,
                    x='colony_size',
                    y='honey_production',
                    color='temperature',
                    size='rainfall',
                    title='Colony Size vs Honey Production',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab2:
            fig_corr = px.imshow(
                st.session_state.data.corr(),
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Matrix'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            # Model comparison metrics
            rf_r2 = r2_score(st.session_state.y_test, 
                           st.session_state.rf_model.predict(st.session_state.X_test))
            lr_r2 = r2_score(st.session_state.y_test, 
                           st.session_state.lr_model.predict(st.session_state.X_test_scaled))
            
            rf_mae = mean_absolute_error(st.session_state.y_test, 
                                       st.session_state.rf_model.predict(st.session_state.X_test))
            lr_mae = mean_absolute_error(st.session_state.y_test, 
                                       st.session_state.lr_model.predict(st.session_state.X_test_scaled))
            
            comparison_data = {
                'Model': ['Random Forest', 'Linear Regression'],
                'R² Score': [rf_r2, lr_r2],
                'Mean Absolute Error': [rf_mae, lr_mae]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance comparison chart
            fig_comparison = px.bar(
                comparison_df, 
                x='Model', 
                y='R² Score',
                title='Model Performance Comparison (R² Score)',
                color='R² Score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🐝 Built with Streamlit | 🤖 Powered by Machine Learning | 🍯 For Better Beekeeping</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()