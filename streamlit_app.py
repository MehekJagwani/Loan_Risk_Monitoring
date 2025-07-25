import streamlit as st
import pandas as pd
import pickle
import io

# Set Streamlit page config
st.set_page_config(page_title="Loan Risk Prediction", layout="wide")

st.title("📊 Loan Risk Prediction")

# Upload file section
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your Excel file here", type=["xlsx"])

# Load model and feature columns
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Function for preprocessing
def preprocess_data(df):
    # EDUCATION (Ordinal)
    education_map = {
        'SSC': 1,
        '12TH': 2,
        'GRADUATE': 3,
        'UNDER GRADUATE': 3,
        'POST-GRADUATE': 4,
        'OTHERS': 1,
        'PROFESSIONAL': 3
    }
    df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna(1).astype(int)

    # One-hot encode other categorical columns
    categorical_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with feature columns used during training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    return df

if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file)
    st.subheader("🔍 Raw Uploaded Data Preview")
    st.dataframe(df_raw.head())

    try:
        df_processed = preprocess_data(df_raw.copy())
        predictions = model.predict(df_processed)

        # Map predictions (0/1/2/3) to P1/P2/etc. if needed
        prediction_labels = [f'P{int(p)+1}' for p in predictions]
        df_raw['Predicted_Risk_Class'] = prediction_labels

        st.subheader("📌 Predicted Risk Classes")
        st.dataframe(df_raw.head())

        # Save to Excel in-memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_raw.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)

        # Download button
        st.download_button(
            label="📥 Download Excel File",
            data=output,
            file_name="loan_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
