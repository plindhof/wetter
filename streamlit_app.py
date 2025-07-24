import streamlit as st
from wetter import DeutscheBahnSentimentAnalyzer

st.set_page_config(page_title="🚆 DB Sentiment-Analyse", layout="wide")

st.title("🚆 Deutsche Bahn Sentiment-Analyse mit KI")
st.markdown("Dieses Tool analysiert simulierte Social-Media-Posts zur DB und zeigt Auswertungen, Trends und Vorhersagen.")

analyzer = DeutscheBahnSentimentAnalyzer()

if st.button("🔄 1. Posts generieren"):
    analyzer.generate_realistic_posts()
    st.success("✅ Posts generiert.")

if st.button("📊 2. Sentiment-Analyse durchführen"):
    df, topics = analyzer.analyze_all_posts()
    st.success("✅ Analyse abgeschlossen.")
    st.dataframe(df.head())

if st.button("🌥️ 3. Wordcloud anzeigen"):
    analyzer.create_wordcloud()

if st.button("📈 4. KI-Trend-Vorhersage"):
    analyzer.predict_sentiment_trends()

if st.button("📋 5. Zusammenfassung anzeigen"):
    analyzer.generate_summary_report()
