from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.header(' ZRP Support Sentiment Analysis App')

# ----- Analyze Individual Text -----
with st.expander('📌 Analyze Text'):
    text = st.text_input('Enter text to analyze:')
    if text:
        blob = TextBlob(text)
        st.write('Polarity:', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity:', round(blob.sentiment.subjectivity, 2))

# ----- Clean Individual Text -----
with st.expander('Clean Text'):
    pre = st.text_input('Enter text to clean:')
    if pre:
        cleaned = cleantext.clean(pre,
                                  clean_all=False,
                                  extra_spaces=True,
                                  stopwords=True,
                                  lowercase=True,
                                  numbers=True,
                                  punct=True)
        st.write('✅ Cleaned Text:', cleaned)

# ----- Analyze CSV -----
with st.expander('Analyze CSV File'):
    upl = st.file_uploader("Upload a CSV file containing tweets/text", type=["csv"])

    # Sentiment scoring function
    def score(x):
        return TextBlob(x).sentiment.polarity

    # Labeling function
    def analyzer(x):
        if x >= 0.5:
            return 'Positive'
        elif x < -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:      # it creates a data frame by uploading
        df = pd.read_csv(upl)

        # Display column names and preview
        st.write("✅ Detected Columns:")
        st.write(df.columns.tolist())
        st.write("📄 Data Preview:")
        st.write(df.head())

        # Try to detect the text column
        possible_cols = [col for col in df.columns if 'tweet' in col.lower() or 'text' in col.lower()]
        if possible_cols:
            text_col = possible_cols[0]  # Use the first matching column
            df['score'] = df[text_col].astype(str).apply(score)
            df['analysis'] = df['score'].apply(analyzer)
            st.success(f"🎯 Sentiment analysis applied to column: '{text_col}'")
            st.write(df[[text_col, 'score', 'analysis']].head())

            @st.cache_data
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)

            st.download_button (
                label="⬇️ Download results as CSV",
                data=csv,
                file_name='sentiment_results.csv',
                mime='text/csv',
            )
        else:
            st.error("❌ No column found with tweet/text data. Please check your CSV.")
