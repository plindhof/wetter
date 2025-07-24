import streamlit as st
import requests

# Streamlit-Layout
st.title("🌤️ Aktuelles Wetter abrufen")
st.subheader("Wähle eine Stadt:")

# Liste möglicher Orte (du kannst sie erweitern)
orte = {
    "Berlin": (52.52, 13.41),
    "Hamburg": (53.55, 10.00),
    "München": (48.14, 11.58),
    "Köln": (50.94, 6.96),
    "Frankfurt": (50.11, 8.68)
}

# Dropdown zur Auswahl
stadt = st.selectbox("Stadt auswählen", list(orte.keys()))
latitude, longitude = orte[stadt]

# API-URL vorbereiten
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={latitude}&longitude={longitude}"
    "&current_weather=true"
)

# API abrufen
try:
    response = requests.get(url)
    data = response.json()

    wetter = data["current_weather"]
    temperatur = wetter["temperature"]
    wind = wetter["windspeed"]

    # Anzeige
    st.success(f"📍 Ort: {stadt}")
    st.metric("🌡️ Temperatur (°C)", temperatur)
    st.metric("💨 Windgeschwindigkeit (km/h)", wind)

except Exception as e:
    st.error("Fehler beim Abrufen der Wetterdaten 😢")
    st.exception(e)
