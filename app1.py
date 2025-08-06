import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from NoCodeTextClassifier.EDA import Informations, Visualizations
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from NoCodeTextClassifier.preprocessing import process, TextCleaner, Vectorization  
from NoCodeTextClassifier.models import Models
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Utility functions
def save_artifacts(obj, folder_name, file_name):
    """Save artifacts like encoders and vectorizers"""
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, file_name), 'wb') as f:
        pickle.dump(obj, f)

def load_artifacts(folder_name, file_name):
    """Load saved artifacts"""
    try:
        with open(os.path.join(folder_name, file_name), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File {file_name} not found in {folder_name} folder")
        return None

def load_model(model_name):
    """Load trained model"""
    try:
        with open(os.path.join('models', model_name), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model {model_name} not found. Please train a model first.")
        return None

def predict_text(model_name, text, vectorizer_type="tfidf"):
    """Make prediction on new text"""
    try:
        # Load model
        model = load_model(model_name)
        if model is None:
            return None, None
        
        # Load vectorizer
        vectorizer_file = f"{vectorizer_type}_vectorizer.pkl"
        vectorizer = load_artifacts("artifacts", vectorizer_file)
        if vectorizer is None:
            return None, None
        
        # Load label encoder
        encoder = load_artifacts("artifacts", "encoder.pkl")
        if encoder is None:
            return None, None
        
        # Clean and vectorize text
        text_cleaner = TextCleaner()
        clean_text = text_cleaner.clean_text(text)
        
        # Transform text using the same vectorizer used during training
        text_vector = vectorizer.transform([clean_text])
        
        # Make prediction
        prediction = model.predict(text_vector)
        prediction_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                prediction_proba = model.predict_proba(text_vector)[0]
            except:
                pass
        
        # Decode prediction
        predicted_label = encoder.inverse_transform(prediction)[0]
        
        return predicted_label, prediction_proba
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Streamlit App
st.title('No Code Text Classification App')
st.write('Understand the behavior of your text data and train a model to classify the text data')

# Sidebar
section = st.sidebar.radio("Choose Section", ["Data Analysis", "Train Model", "Predictions"])

# Upload Data
st.sidebar.subheader("Upload Your Dataset")
train_data = st.sidebar.file_uploader("Upload training data", type=["csv"])
test_data = st.sidebar.file_uploader("Upload test data (optional)", type=["csv"])

# Global variables to store data and settings
if 'vectorizer_type' not in st.session_state:
    st.session_state.vectorizer_type = "tfidf"

if train_data is not None:
    try:
        train_df = pd.read_csv(train_data, encoding='latin1')
        
        if test_data is not None:
            test_df = pd.read_csv(test_data, encoding='latin1')
        else:
            test_df = None
            
        st.write("Training Data Preview:")
        st.write(train_df.head(3))
        
        columns = train_df.columns.tolist()
        text_data = st.sidebar.selectbox("Choose the text column:", columns)
        target = st.sidebar.selectbox("Choose the target column:", columns)

        # Process data
        info = Informations(train_df, text_data, target)
        train_df['clean_text'] = info.clean_text()
        train_df['text_length'] = info.text_length()
        
        # Handle label encoding manually if the class doesn't store encoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        train_df['target'] = label_encoder.fit_transform(train_df[target])
        
        # Save label encoder for later use
        os.makedirs("artifacts", exist_ok=True)
        save_artifacts(label_encoder, "artifacts", "encoder.pkl")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        train_df = None
        info = None

# Data Analysis Section
if section == "Data Analysis":
    if train_data is not None and train_df is not None:
        try:
            st.subheader("Get Insights from the Data")
            
            st.write("Data Shape:", info.shape())
            st.write("Class Imbalance:", info.class_imbalanced())
            st.write("Missing Values:", info.missing_values())

            st.write("Processed Data Preview:")
            st.write(train_df[['clean_text', 'text_length', 'target']].head(3))
            
            st.markdown("**Text Length Analysis**")
            st.write(info.analysis_text_length('text_length'))
            
            # Calculate correlation manually since we handled encoding separately
            correlation = train_df[['text_length', 'target']].corr().iloc[0, 1]
            st.write(f"Correlation between Text Length and Target: {correlation:.4f}")

            st.subheader("Visualizations")
            vis = Visualizations(train_df, text_data, target)
            vis.class_distribution()
            vis.text_length_distribution()

        except Exception as e:
            st.error(f"Error in data analysis: {str(e)}")
    else:
        st.warning("Please upload training data to get insights")

# Train Model Section
elif section == "Train Model":
    if train_data is not None and train_df is not None:
        try:
            st.subheader("Train a Model")

            # Create two columns for model selection
            col1, col2 = st.columns(2)

            with col1:
                model = st.radio("Choose the Model", [
                    "Logistic Regression", "Decision Tree", 
                    "Random Forest", "Linear SVC", "SVC",
                    "Multinomial Naive Bayes", "Gaussian Naive Bayes"
                ])
            
            with col2:
                vectorizer_choice = st.radio("Choose Vectorizer", ["Tfidf Vectorizer", "Count Vectorizer"])

            # Initialize vectorizer
            if vectorizer_choice == "Tfidf Vectorizer":
                vectorizer = TfidfVectorizer(max_features=10000)
                st.session_state.vectorizer_type = "tfidf"
            else:
                vectorizer = CountVectorizer(max_features=10000)
                st.session_state.vectorizer_type = "count"

            st.write("Training Data Preview:")
            st.write(train_df[['clean_text', 'target']].head(3))
            
            # Vectorize text data
            X = vectorizer.fit_transform(train_df['clean_text'])
            y = train_df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = process.split_data(X, y)
            st.write(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Save vectorizer for later use
            vectorizer_filename = f"{st.session_state.vectorizer_type}_vectorizer.pkl"
            save_artifacts(vectorizer, "artifacts", vectorizer_filename)
            
            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    models = Models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
                    
                    # Train selected model
                    if model == "Logistic Regression":
                        models.LogisticRegression()
                    elif model == "Decision Tree":
                        models.DecisionTree()
                    elif model == "Linear SVC":
                        models.LinearSVC()
                    elif model == "SVC":
                        models.SVC()
                    elif model == "Multinomial Naive Bayes":
                        models.MultinomialNB()
                    elif model == "Random Forest":
                        models.RandomForestClassifier()
                    elif model == "Gaussian Naive Bayes":
                        models.GaussianNB()
                
                st.success("Model training completed!")
                st.info("You can now use the 'Predictions' section to classify new text.")

        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
    else:
        st.warning("Please upload training data to train a model")

# Predictions Section
elif section == "Predictions":
    st.subheader("Perform Predictions on New Text")
    
    # Check if models exist
    if os.path.exists("models") and os.listdir("models"):
        # Text input for prediction
        text_input = st.text_area("Enter the text to classify:", height=100)
        
        # Model selection
        available_models = [f for f in os.listdir("models") if f.endswith('.pkl')]
        
        if available_models:
            selected_model = st.selectbox("Choose the trained model:", available_models)
            
            # Prediction button
            if st.button("Predict", key="single_predict"):
                if text_input.strip():
                    with st.spinner("Making prediction..."):
                        predicted_label, prediction_proba = predict_text(
                            selected_model, 
                            text_input, 
                            st.session_state.get('vectorizer_type', 'tfidf')
                        )
                        
                        if predicted_label is not None:
                            st.success("Prediction completed!")
                            
                            # Display results
                            st.markdown("### Prediction Results")
                            st.markdown(f"**Input Text:** {text_input}")
                            st.markdown(f"**Predicted Class:** {predicted_label}")
                            
                            # Display probabilities if available
                            if prediction_proba is not None:
                                st.markdown("**Class Probabilities:**")
                                
                                # Load encoder to get class names
                                encoder = load_artifacts("artifacts", "encoder.pkl")
                                if encoder is not None:
                                    classes = encoder.classes_
                                    prob_df = pd.DataFrame({
                                        'Class': classes,
                                        'Probability': prediction_proba
                                    }).sort_values('Probability', ascending=False)
                                    
                                    st.bar_chart(prob_df.set_index('Class'))
                                    st.dataframe(prob_df)
                else:
                    st.warning("Please enter some text to classify")
        else:
            st.warning("No trained models found. Please train a model first.")
    else:
        st.warning("No trained models found. Please go to 'Train Model' section to train a model first.")
        
    # Option to classify multiple texts
    st.markdown("---")
    st.subheader("Batch Predictions")
    
    uploaded_file = st.file_uploader("Upload a CSV file with text to classify", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, encoding='latin1')
            st.write("Uploaded data preview:")
            st.write(batch_df.head())
            
            # Select text column
            text_column = st.selectbox("Select the text column:", batch_df.columns.tolist())
            
            if os.path.exists("models") and os.listdir("models"):
                available_models = [f for f in os.listdir("models") if f.endswith('.pkl')]
                batch_model = st.selectbox("Choose model for batch prediction:", available_models, key="batch_model")
                
                if st.button("Run Batch Predictions", key="batch_predict"):
                    with st.spinner("Processing batch predictions..."):
                        predictions = []
                        
                        for text in batch_df[text_column]:
                            pred, _ = predict_text(
                                batch_model, 
                                str(text), 
                                st.session_state.get('vectorizer_type', 'tfidf')
                            )
                            predictions.append(pred if pred is not None else "Error")
                        
                        batch_df['Predicted_Class'] = predictions
                        
                        st.success("Batch predictions completed!")
                        st.write("Results:")
                        st.write(batch_df[[text_column, 'Predicted_Class']])
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download predictions as CSV",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error in batch prediction: {str(e)}")
