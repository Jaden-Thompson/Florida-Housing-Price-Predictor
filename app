import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Florida Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

def load_data():
    """Load and preprocess the Florida housing price data"""
    try:
        # Load the data
        df = pd.read_csv('attached_assets/FLSTHPI_1750271998897.csv')
        
        # Convert observation_date to datetime
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        
        # Remove any rows with missing data
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def engineer_features(df):
    """Create time-based features for the regression model"""
    # Create a copy to avoid modifying original data
    df_features = df.copy()
    
    # Extract temporal features
    df_features['year'] = df_features['observation_date'].dt.year
    df_features['quarter'] = df_features['observation_date'].dt.quarter
    df_features['month'] = df_features['observation_date'].dt.month
    
    # Create cyclical features for seasonality
    df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
    df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    
    # Create trend features
    df_features['days_since_start'] = (df_features['observation_date'] - df_features['observation_date'].min()).dt.days
    df_features['years_since_start'] = df_features['days_since_start'] / 365.25
    
    # Create lagged features (if enough data points)
    if len(df_features) > 4:
        df_features['price_lag_1'] = df_features['FLSTHPI'].shift(1)
        df_features['price_lag_4'] = df_features['FLSTHPI'].shift(4)  # Year-over-year
    
    # Calculate moving averages
    if len(df_features) > 8:
        df_features['price_ma_4'] = df_features['FLSTHPI'].rolling(window=4, min_periods=1).mean()
        df_features['price_ma_8'] = df_features['FLSTHPI'].rolling(window=8, min_periods=1).mean()
    
    # Calculate price changes
    df_features['price_change'] = df_features['FLSTHPI'].pct_change()
    df_features['price_change_4'] = df_features['FLSTHPI'].pct_change(periods=4)
    
    # Fill NaN values created by lagged features and changes
    df_features = df_features.bfill().ffill()
    
    return df_features

def train_ridge_model(df_features, test_size=0.2, random_state=42):
    """Train Ridge Regression model with proper validation"""
    
    # Define feature columns (excluding target and date)
    feature_cols = [col for col in df_features.columns 
                   if col not in ['observation_date', 'FLSTHPI']]
    
    X = df_features[feature_cols]
    y = df_features['FLSTHPI']
    
    # Split data temporally (use last 20% as test set)
    split_idx = int(len(df_features) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge regression with cross-validation for alpha selection
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha = 1.0
    best_cv_score = -np.inf
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=random_state)
        cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2')
        avg_cv_score = np.mean(cv_scores)
        
        if avg_cv_score > best_cv_score:
            best_cv_score = avg_cv_score
            best_alpha = alpha
    
    # Train final model with best alpha
    final_model = Ridge(alpha=best_alpha, random_state=random_state)
    final_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = final_model.predict(X_train_scaled)
    y_test_pred = final_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results = {
        'model': final_model,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'best_cv_score': best_cv_score,
        'feature_cols': feature_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'split_idx': split_idx
    }
    
    return results

def generate_future_predictions(df_features, model_results, periods=20):
    """Generate predictions for future periods"""
    
    # Get the last observation date
    last_date = df_features['observation_date'].max()
    
    # Create future dates (quarterly data)
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=3),
        periods=periods,
        freq='Q'
    )
    
    # Create future dataframe
    future_df = pd.DataFrame({'observation_date': future_dates})
    
    # Engineer features for future dates
    future_df['year'] = future_df['observation_date'].dt.year
    future_df['quarter'] = future_df['observation_date'].dt.quarter
    future_df['month'] = future_df['observation_date'].dt.month
    
    # Cyclical features
    future_df['quarter_sin'] = np.sin(2 * np.pi * future_df['quarter'] / 4)
    future_df['quarter_cos'] = np.cos(2 * np.pi * future_df['quarter'] / 4)
    
    # Trend features
    start_date = df_features['observation_date'].min()
    future_df['days_since_start'] = (future_df['observation_date'] - start_date).dt.days
    future_df['years_since_start'] = future_df['days_since_start'] / 365.25
    
    # For lagged features, use the last known values and predictions
    last_price = df_features['FLSTHPI'].iloc[-1]
    last_4_price = df_features['FLSTHPI'].iloc[-4] if len(df_features) >= 4 else last_price
    last_ma_4 = df_features['price_ma_4'].iloc[-1] if 'price_ma_4' in df_features.columns else last_price
    last_ma_8 = df_features['price_ma_8'].iloc[-1] if 'price_ma_8' in df_features.columns else last_price
    
    # Initialize lagged features (simplified approach)
    future_df['price_lag_1'] = last_price
    future_df['price_lag_4'] = last_4_price
    future_df['price_ma_4'] = last_ma_4
    future_df['price_ma_8'] = last_ma_8
    future_df['price_change'] = 0.01  # Small positive change assumption
    future_df['price_change_4'] = 0.05  # Annual change assumption
    
    # Ensure all feature columns are present
    for col in model_results['feature_cols']:
        if col not in future_df.columns:
            future_df[col] = 0
    
    # Select features in the same order as training
    X_future = future_df[model_results['feature_cols']]
    
    # Scale features
    X_future_scaled = model_results['scaler'].transform(X_future)
    
    # Make predictions
    future_predictions = model_results['model'].predict(X_future_scaled)
    
    # Create results dataframe
    future_results = pd.DataFrame({
        'observation_date': future_dates,
        'predicted_FLSTHPI': future_predictions
    })
    
    return future_results

def main():
    st.title("🏠 Florida Housing Price Predictor")
    st.markdown("### Ridge Regression Analysis of Florida Housing Market Trends (1975-2025)")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Unable to load data. Please check the data file.")
        return
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", value=42, min_value=0)
    future_periods = st.sidebar.slider("Future Prediction Periods", 4, 40, 20, 4)
    
    # Display basic data info
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Date Range", f"{df['observation_date'].min().year}-{df['observation_date'].max().year}")
    with col3:
        st.metric("Min Price Index", f"{df['FLSTHPI'].min():.2f}")
    with col4:
        st.metric("Max Price Index", f"{df['FLSTHPI'].max():.2f}")
    
    # Historical data visualization
    st.header("📈 Historical Housing Price Trends")
    
    fig_hist = px.line(df, x='observation_date', y='FLSTHPI',
                       title='Florida Housing Price Index Over Time',
                       labels={'FLSTHPI': 'Housing Price Index', 'observation_date': 'Date'})
    fig_hist.update_layout(height=500)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Feature engineering
    st.header("🔧 Feature Engineering")
    with st.spinner("Engineering features..."):
        df_features = engineer_features(df)
    
    st.success(f"Created {len(df_features.columns) - 2} features from temporal data")
    
    # Display feature correlation
    if st.checkbox("Show Feature Correlations"):
        feature_cols = [col for col in df_features.columns 
                       if col not in ['observation_date', 'FLSTHPI']]
        if len(feature_cols) > 0:
            try:
                corr_data = df_features[feature_cols + ['FLSTHPI']].corr()['FLSTHPI'].drop('FLSTHPI')
                # Create correlation dataframe for plotting
                corr_df = pd.DataFrame({
                    'feature': corr_data.index,
                    'correlation': corr_data.values
                })
                corr_df['abs_correlation'] = corr_df['correlation'].abs()
                corr_df = corr_df.sort_values('abs_correlation', ascending=False)
                
                fig_corr = px.bar(corr_df, x='correlation', y='feature',
                                 title='Feature Correlation with Housing Price Index',
                                 labels={'correlation': 'Correlation', 'feature': 'Features'},
                                 orientation='h')
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display correlation matrix: {str(e)}")
        else:
            st.info("No features available for correlation analysis.")
    
    # Model training
    st.header("🤖 Model Training & Evaluation")
    
    with st.spinner("Training Ridge Regression model..."):
        model_results = train_ridge_model(df_features, test_size=test_size, random_state=random_state)
    
    # Display model performance
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Set Performance**")
        st.metric("R² Score", f"{model_results['train_r2']:.4f}")
        st.metric("MAE", f"{model_results['train_mae']:.2f}")
        st.metric("RMSE", f"{model_results['train_rmse']:.2f}")
    
    with col2:
        st.markdown("**Test Set Performance**")
        st.metric("R² Score", f"{model_results['test_r2']:.4f}")
        st.metric("MAE", f"{model_results['test_mae']:.2f}")
        st.metric("RMSE", f"{model_results['test_rmse']:.2f}")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Best Alpha (Ridge)", f"{model_results['best_alpha']}")
    with col4:
        st.metric("CV Score", f"{model_results['best_cv_score']:.4f}")
    
    # Model predictions visualization
    st.subheader("Model Predictions vs Actual Values")
    
    # Create combined dataframe for plotting
    train_dates = df_features['observation_date'].iloc[:model_results['split_idx']]
    test_dates = df_features['observation_date'].iloc[model_results['split_idx']:]
    
    fig_pred = go.Figure()
    
    # Historical data
    fig_pred.add_trace(go.Scatter(
        x=df_features['observation_date'],
        y=df_features['FLSTHPI'],
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue', width=2)
    ))
    
    # Training predictions
    fig_pred.add_trace(go.Scatter(
        x=train_dates,
        y=model_results['y_train_pred'],
        mode='lines',
        name='Training Predictions',
        line=dict(color='green', width=2, dash='dot')
    ))
    
    # Test predictions
    fig_pred.add_trace(go.Scatter(
        x=test_dates,
        y=model_results['y_test_pred'],
        mode='lines',
        name='Test Predictions',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add annotation for train/test split
    split_date = train_dates.iloc[-1]
    fig_pred.add_annotation(
        x=split_date,
        y=df_features['FLSTHPI'].max() * 0.9,
        text="Train/Test Split",
        showarrow=True,
        arrowhead=2,
        arrowcolor="gray",
        bgcolor="white",
        bordercolor="gray"
    )
    
    fig_pred.update_layout(
        title='Model Predictions vs Actual Housing Prices',
        xaxis_title='Date',
        yaxis_title='Housing Price Index',
        height=500
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Future predictions
    st.header("🔮 Future Price Predictions")
    
    with st.spinner("Generating future predictions..."):
        future_results = generate_future_predictions(df_features, model_results, future_periods)
    
    # Display future predictions
    st.subheader(f"Predicted Housing Prices for Next {future_periods} Quarters")
    
    # Combine historical and future data for visualization
    fig_future = go.Figure()
    
    # Historical data
    fig_future.add_trace(go.Scatter(
        x=df_features['observation_date'],
        y=df_features['FLSTHPI'],
        mode='lines',
        name='Historical Prices',
        line=dict(color='blue', width=2)
    ))
    
    # Future predictions
    fig_future.add_trace(go.Scatter(
        x=future_results['observation_date'],
        y=future_results['predicted_FLSTHPI'],
        mode='lines+markers',
        name='Future Predictions',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add annotation for historical/future split
    split_date = df_features['observation_date'].max()
    max_price = max(df_features['FLSTHPI'].max(), future_results['predicted_FLSTHPI'].max())
    fig_future.add_annotation(
        x=split_date,
        y=max_price * 0.9,
        text="Historical/Future Split",
        showarrow=True,
        arrowhead=2,
        arrowcolor="gray",
        bgcolor="white",
        bordercolor="gray"
    )
    
    fig_future.update_layout(
        title='Historical Prices and Future Predictions',
        xaxis_title='Date',
        yaxis_title='Housing Price Index',
        height=500
    )
    
    st.plotly_chart(fig_future, use_container_width=True)
    
    # Future predictions table
    st.subheader("Future Predictions Table")
    
    # Format future results for display
    future_display = future_results.copy()
    future_display['observation_date'] = future_display['observation_date'].dt.strftime('%Y-%m-%d')
    future_display['predicted_FLSTHPI'] = future_display['predicted_FLSTHPI'].round(2)
    future_display = future_display.rename(columns={
        'observation_date': 'Date',
        'predicted_FLSTHPI': 'Predicted Price Index'
    })
    
    st.dataframe(future_display, use_container_width=True)
    
    # Market analysis
    st.header("📊 Market Analysis")
    
    # Calculate trends
    recent_trend = (df['FLSTHPI'].iloc[-1] - df['FLSTHPI'].iloc[-5]) / df['FLSTHPI'].iloc[-5] * 100
    future_trend = (future_results['predicted_FLSTHPI'].iloc[-1] - future_results['predicted_FLSTHPI'].iloc[0]) / future_results['predicted_FLSTHPI'].iloc[0] * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Recent Trend (Last Year)",
            f"{recent_trend:.2f}%",
            delta=f"{recent_trend:.2f}%"
        )
    
    with col2:
        st.metric(
            "Predicted Future Trend",
            f"{future_trend:.2f}%",
            delta=f"{future_trend:.2f}%"
        )
    
    with col3:
        current_price = df['FLSTHPI'].iloc[-1]
        predicted_next = future_results['predicted_FLSTHPI'].iloc[0]
        next_change = (predicted_next - current_price) / current_price * 100
        st.metric(
            "Next Quarter Change",
            f"{next_change:.2f}%",
            delta=f"{next_change:.2f}%"
        )
    
    # Model explanation
    st.header("🧠 How the Ridge Regression Model Works")
    
    with st.expander("Click to learn about Ridge Regression and Feature Engineering"):
        st.markdown("""
        ### Ridge Regression Explained
        
        **What is Ridge Regression?**
        - Ridge regression is a type of linear regression that adds regularization to prevent overfitting
        - It adds a penalty term (L2 regularization) to the standard linear regression cost function
        - The penalty term is controlled by the alpha parameter (α) - higher α means more regularization
        
        **Mathematical Formula:**
        ```
        Cost = MSE + α × (sum of squared coefficients)
        ```
        
        **Why Ridge Regression for Housing Prices?**
        1. **Handles Multicollinearity**: When features are correlated (like year and time trends)
        2. **Prevents Overfitting**: Regularization keeps the model from memorizing training data
        3. **Stable Predictions**: Works well with time series data that has seasonal patterns
        4. **Feature Selection**: Shrinks less important feature coefficients toward zero
        
        ### Feature Engineering Process
        
        **Temporal Features:**
        - **Year, Quarter, Month**: Direct time components
        - **Days/Years since start**: Linear time trend
        
        **Cyclical Features:**
        - **Quarter sine/cosine**: Captures seasonal patterns mathematically
        - **Why sine/cosine?** These functions repeat every 4 quarters, perfect for seasonality
        
        **Price History Features:**
        - **Lagged prices**: Previous quarter's price (price_lag_1)
        - **Year-over-year**: Price from 4 quarters ago (price_lag_4)
        - **Moving averages**: Smoothed price trends (4 and 8 quarter windows)
        - **Price changes**: Percentage changes over time
        
        ### Model Training Pipeline
        
        1. **Data Split**: Temporal split (80% training, 20% testing) - no random shuffling
        2. **Feature Scaling**: StandardScaler normalizes all features to same scale
        3. **Hyperparameter Tuning**: Cross-validation finds best alpha value
        4. **Model Training**: Ridge regression learns feature weights
        5. **Validation**: Test on unseen recent data to measure real performance
        
        ### Performance Metrics
        
        - **R² Score**: Percentage of price variance explained by the model
        - **MAE (Mean Absolute Error)**: Average prediction error in price index points
        - **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
        """)
    
    # Model insights
    st.header("💡 Model Insights")
    
    st.markdown(f"""
    **Model Summary:**
    - **Algorithm**: Ridge Regression with regularization parameter α = {model_results['best_alpha']}
    - **Features**: {len(model_results['feature_cols'])} engineered features including temporal patterns, seasonality, and price history
    - **Training Period**: {df['observation_date'].min().strftime('%Y-%m-%d')} to {df_features['observation_date'].iloc[model_results['split_idx']-1].strftime('%Y-%m-%d')}
    - **Test Period**: {df_features['observation_date'].iloc[model_results['split_idx']].strftime('%Y-%m-%d')} to {df['observation_date'].max().strftime('%Y-%m-%d')}
    - **Model Performance**: R² = {model_results['test_r2']:.4f} on test set
    
    **Key Findings:**
    - The model achieved an R² score of {model_results['test_r2']:.4f}, indicating {model_results['test_r2']*100:.1f}% of price variance is explained
    - Recent market trend shows a {recent_trend:.2f}% change over the last year
    - Future predictions suggest a {future_trend:.2f}% trend over the next {future_periods} quarters
    - The model uses temporal features and price history to capture seasonal patterns and long-term trends
    """)
    
    # Feature importance (Ridge coefficients)
    if st.checkbox("Show Feature Importance"):
        feature_importance = pd.DataFrame({
            'feature': model_results['feature_cols'],
            'coefficient': model_results['model'].coef_
        })
        feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        fig_importance = px.bar(
            feature_importance.head(10),
            x='abs_coefficient',
            y='feature',
            title='Top 10 Most Important Features (Ridge Coefficients)',
            labels={'abs_coefficient': 'Absolute Coefficient Value', 'feature': 'Features'},
            orientation='h'
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Code viewer section
    st.header("💻 View the Source Code")
    
    code_section = st.selectbox(
        "Select code section to view:",
        ["Complete App Code", "Data Loading", "Feature Engineering", "Ridge Model Training", "Future Predictions"]
    )
    
    if code_section == "Complete App Code":
        with st.expander("Complete Application Code", expanded=False):
            with open(__file__, 'r') as f:
                code_content = f.read()
            st.code(code_content, language='python')
    
    elif code_section == "Data Loading":
        with st.expander("Data Loading Function", expanded=True):
            st.code('''
def load_data():
    """Load and preprocess the Florida housing price data"""
    try:
        # Load the CSV file containing Florida STHPI data
        df = pd.read_csv('attached_assets/FLSTHPI_1750271998897.csv')
        
        # Convert observation_date to datetime for time series analysis
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        
        # Remove any rows with missing data to ensure clean dataset
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
            ''', language='python')
    
    elif code_section == "Feature Engineering":
        with st.expander("Feature Engineering Function", expanded=True):
            st.code('''
def engineer_features(df):
    """Create time-based features for the regression model"""
    # Create a copy to avoid modifying original data
    df_features = df.copy()
    
    # Extract basic temporal features
    df_features['year'] = df_features['observation_date'].dt.year
    df_features['quarter'] = df_features['observation_date'].dt.quarter
    df_features['month'] = df_features['observation_date'].dt.month
    
    # Create cyclical features for seasonality (sine/cosine transformation)
    # This captures the cyclical nature of quarters (Q1 is similar to next Q1)
    df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
    df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    
    # Create trend features (time progression)
    df_features['days_since_start'] = (df_features['observation_date'] - df_features['observation_date'].min()).dt.days
    df_features['years_since_start'] = df_features['days_since_start'] / 365.25
    
    # Create lagged features (previous values)
    if len(df_features) > 4:
        df_features['price_lag_1'] = df_features['FLSTHPI'].shift(1)  # Previous quarter
        df_features['price_lag_4'] = df_features['FLSTHPI'].shift(4)  # Same quarter last year
    
    # Calculate moving averages (trend smoothing)
    if len(df_features) > 8:
        df_features['price_ma_4'] = df_features['FLSTHPI'].rolling(window=4, min_periods=1).mean()
        df_features['price_ma_8'] = df_features['FLSTHPI'].rolling(window=8, min_periods=1).mean()
    
    # Calculate price changes (momentum indicators)
    df_features['price_change'] = df_features['FLSTHPI'].pct_change()
    df_features['price_change_4'] = df_features['FLSTHPI'].pct_change(periods=4)
    
    # Fill NaN values created by lagged features and changes
    df_features = df_features.bfill().ffill()
    
    return df_features
            ''', language='python')
    
    elif code_section == "Ridge Model Training":
        with st.expander("Ridge Regression Training Function", expanded=True):
            st.code('''
def train_ridge_model(df_features, test_size=0.2, random_state=42):
    """Train Ridge Regression model with proper validation"""
    
    # Define feature columns (exclude target variable and date)
    feature_cols = [col for col in df_features.columns 
                   if col not in ['observation_date', 'FLSTHPI']]
    
    X = df_features[feature_cols]  # Features
    y = df_features['FLSTHPI']     # Target variable
    
    # Temporal split (important: no random shuffling for time series)
    split_idx = int(len(df_features) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Feature scaling (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning using cross-validation
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]  # Regularization strengths
    best_alpha = 1.0
    best_cv_score = -np.inf
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=random_state)
        cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2')
        avg_cv_score = np.mean(cv_scores)
        
        if avg_cv_score > best_cv_score:
            best_cv_score = avg_cv_score
            best_alpha = alpha
    
    # Train final model with best alpha
    final_model = Ridge(alpha=best_alpha, random_state=random_state)
    final_model.fit(X_train_scaled, y_train)
    
    # Make predictions and calculate performance metrics
    y_train_pred = final_model.predict(X_train_scaled)
    y_test_pred = final_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return {
        'model': final_model, 'scaler': scaler, 'best_alpha': best_alpha,
        'best_cv_score': best_cv_score, 'feature_cols': feature_cols,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_rmse': train_rmse, 'test_rmse': test_rmse
        # ... other results
    }
            ''', language='python')
    
    elif code_section == "Future Predictions":
        with st.expander("Future Predictions Function", expanded=True):
            st.code('''
def generate_future_predictions(df_features, model_results, periods=20):
    """Generate predictions for future periods"""
    
    # Get the last observation date from historical data
    last_date = df_features['observation_date'].max()
    
    # Create future dates (quarterly frequency)
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=3),
        periods=periods,
        freq='Q'  # Quarterly frequency
    )
    
    # Create future dataframe with same feature engineering
    future_df = pd.DataFrame({'observation_date': future_dates})
    
    # Engineer features for future dates (same as historical data)
    future_df['year'] = future_df['observation_date'].dt.year
    future_df['quarter'] = future_df['observation_date'].dt.quarter
    future_df['month'] = future_df['observation_date'].dt.month
    
    # Cyclical features
    future_df['quarter_sin'] = np.sin(2 * np.pi * future_df['quarter'] / 4)
    future_df['quarter_cos'] = np.cos(2 * np.pi * future_df['quarter'] / 4)
    
    # Trend features
    start_date = df_features['observation_date'].min()
    future_df['days_since_start'] = (future_df['observation_date'] - start_date).dt.days
    future_df['years_since_start'] = future_df['days_since_start'] / 365.25
    
    # For lagged features, use last known values (simplified approach)
    last_price = df_features['FLSTHPI'].iloc[-1]
    future_df['price_lag_1'] = last_price
    future_df['price_lag_4'] = df_features['FLSTHPI'].iloc[-4] if len(df_features) >= 4 else last_price
    # ... initialize other lagged features
    
    # Apply same feature scaling
    X_future = future_df[model_results['feature_cols']]
    X_future_scaled = model_results['scaler'].transform(X_future)
    
    # Generate predictions using trained model
    future_predictions = model_results['model'].predict(X_future_scaled)
    
    return pd.DataFrame({
        'observation_date': future_dates,
        'predicted_FLSTHPI': future_predictions
    })
            ''', language='python')

if __name__ == "__main__":
    main()
