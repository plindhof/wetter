# Deutsche Bahn Sentiment-Analyse - KI-Projekt
# KI-gestützte Analyse der öffentlichen Meinung zur Deutschen Bahn

# Installiere benötigte Bibliotheken (in Google Colab ausführen)
# !pip install textblob matplotlib seaborn wordcloud pandas plotly scikit-learn numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from collections import Counter
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Deutsche Konfiguration
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class DeutscheBahnSentimentAnalyzer:
    def __init__(self):
        self.posts = []
        self.analyzed_data = None

        # Deutsche Negationswörter für bessere Sentiment-Analyse
        self.negation_words = ['nicht', 'kein', 'keine', 'niemals', 'nie', 'nichts', 'ohne']

        # Deutsche positive/negative Wörter für Sentiment-Verbesserung
        self.positive_words = ['gut', 'super', 'toll', 'perfekt', 'hervorragend', 'pünktlich',
                              'sauber', 'freundlich', 'schnell', 'bequem', 'günstig']
        self.negative_words = ['schlecht', 'schlimm', 'katastrophe', 'verspätung', 'ausfall',
                              'teuer', 'dreckig', 'unfreundlich', 'langsam', 'überfüllt', 'chaos']

    def generate_realistic_posts(self):
        """Generiert realistische Social Media Posts über die Deutsche Bahn"""

        # Realistische Post-Templates mit verschiedenen Sentiment-Ausprägungen
        negative_posts = [
            "Schon wieder 20 Minuten Verspätung bei der {strecke}. Das ist echt nicht mehr normal! #DB #Verspätung",
            "Der Zug fällt komplett aus und die nächste Verbindung ist erst in 2 Stunden. Danke für nichts, Deutsche Bahn! 😡",
            "Überfüllter Zug, kaputte Klimaanlage und dann noch Verspätung. Für diese Preise eine Frechheit! #DBFail",
            "Wann lernt die DB endlich, dass man Züge nicht einfach streichen kann ohne Alternative? Stehe seit 1h am Bahnhof 🙄",
            "45 Minuten Verspätung wegen 'technischer Störung'. Was läuft da eigentlich schief bei der Deutschen Bahn?",
            "Ticket für 89€ gekauft und dann steht man im Gang weil keine Reservierung möglich war. Lächerlich!",
            "Die DB App zeigt was anderes an als die Anzeigetafel. Chaos pur! Wie soll man da planen?",
            "Schaffner war heute richtig unfreundlich. Service wird bei der DB wohl klein geschrieben 😤",
            "Toiletten defekt, WLAN funktioniert nicht, Bordrestaurant geschlossen. Toller Service für den Preis!",
            "3. Ausfall diese Woche auf derselben Strecke. Die DB ist einfach nicht zuverlässig 😠"
        ]

        neutral_posts = [
            "Fahre heute mit der {strecke}. Mal schauen wie pünktlich wir ankommen 🤞",
            "Bahnfahren hat schon was. Kann dabei arbeiten und muss nicht selbst fahren 🚆",
            "Die neuen ICE-Züge sehen schon schick aus. Bin gespannt auf die Fahrt",
            "Pendeln mit der Bahn ist günstiger als Auto fahren, auch wenn's manchmal nerve",
            "Bahnhof {stadt} wurde schön renoviert. Sieht jetzt viel moderner aus",
            "Die DB Connect App hat ein Update bekommen. Sieht übersichtlicher aus",
            "Fahrkartenautomat war heute mal nicht defekt. Kleine Wunder gibt es doch 😄",
            "Im Zug ist es wenigstens ruhiger als im Flugzeug. Kann entspannt lesen",
            "Deutsche Bahn investiert in neue Züge. Hoffentlich wird's dadurch besser",
            "Zugfahren ist definitiv nachhaltiger als fliegen. Gut für die Umwelt 🌱"
        ]

        positive_posts = [
            "Heute mal pünktlich angekommen! Danke DB, geht doch! 👍 #BahnFährtPünktlich",
            "Super freundlicher Schaffner heute. Hat mir sogar beim Umsteigen geholfen! 😊",
            "Der neue ICE 4 ist wirklich komfortabel. WLAN funktioniert auch gut 📶",
            "Fahrt von {stadt} nach {stadt} war entspannt und pünktlich. So soll es sein! ✨",
            "Bordrestaurant hatte leckeren Kaffee und das Personal war sehr nett ☕",
            "Spontan noch ein Sparpreis-Ticket bekommen. 29€ quer durch Deutschland ist schon fair!",
            "Die Sitze im ICE sind bequemer als gedacht. Kann gut arbeiten während der Fahrt 💻",
            "Bahnfahren ist schon entspannend. Kann die Landschaft genießen statt Autobahn 🌲",
            "Pünktliche Ankunft und sauberer Zug. Die DB kann es also doch! 🎉",
            "Gut dass es die Bahn gibt. Ohne Auto trotzdem mobil sein 🚉"
        ]

        strecken = ["ICE nach München", "RE nach Hamburg", "S-Bahn Richtung Flughafen", "ICE nach Berlin",
                   "RB nach Frankfurt", "IC nach Köln", "Regionalzug", "Fernzug"]
        städte = ["Berlin", "München", "Hamburg", "Köln", "Frankfurt", "Stuttgart", "Düsseldorf", "Leipzig"]
        plattformen = ["Twitter", "Facebook", "Reddit", "Instagram"]

        posts = []

        # Generiere negative Posts (60%)
        for i in range(90):
            post = np.random.choice(negative_posts)
            post = post.format(strecke=np.random.choice(strecken),
                             stadt=np.random.choice(städte))
            posts.append({
                'text': post,
                'platform': np.random.choice(plattformen),
                'date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'expected_sentiment': 'negativ'
            })

        # Generiere neutrale Posts (25%)
        for i in range(40):
            post = np.random.choice(neutral_posts)
            post = post.format(strecke=np.random.choice(strecken),
                             stadt=np.random.choice(städte))
            posts.append({
                'text': post,
                'platform': np.random.choice(plattformen),
                'date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'expected_sentiment': 'neutral'
            })

        # Generiere positive Posts (15%)
        for i in range(20):
            post = np.random.choice(positive_posts)
            post = post.format(strecke=np.random.choice(strecken),
                             stadt=np.random.choice(städte))
            posts.append({
                'text': post,
                'platform': np.random.choice(plattformen),
                'date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'expected_sentiment': 'positiv'
            })

        self.posts = posts
        print(f"✅ {len(posts)} realistische Deutsche Bahn Posts generiert")
        return posts

    def analyze_sentiment_german(self, text):
        """Erweiterte deutsche Sentiment-Analyse mit Negationserkennung"""

        # Basis-Sentiment mit TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        # Deutsche Wörter-basierte Verbesserung
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)

        # Negationserkennung
        has_negation = any(neg in text_lower for neg in self.negation_words)
        if has_negation:
            polarity *= -0.5  # Abschwächung bei Negation

        # Sentiment-Score anpassen basierend auf deutschen Wörtern
        if positive_count > negative_count:
            polarity += 0.3
        elif negative_count > positive_count:
            polarity -= 0.3

        # Emojis berücksichtigen
        if '😊' in text or '👍' in text or '✨' in text or '🎉' in text:
            polarity += 0.2
        if '😡' in text or '😠' in text or '🙄' in text or '😤' in text:
            polarity -= 0.3

        # Klassifizierung
        if polarity > 0.1:
            return 'positiv', polarity
        elif polarity < -0.1:
            return 'negativ', polarity
        else:
            return 'neutral', polarity

    def extract_topics(self, texts):
        """Extrahiert Hauptthemen aus den Posts"""

        topic_keywords = {
            'Verspätungen': ['verspätung', 'verspätet', 'zu spät', 'unpünktlich', 'delay'],
            'Ausfälle': ['ausfall', 'gestrichen', 'fällt aus', 'entfällt', 'cancelled'],
            'Service': ['service', 'schaffner', 'personal', 'freundlich', 'unfreundlich', 'bedienung'],
            'Preise': ['preis', 'teuer', 'günstig', 'ticket', 'euro', '€', 'kosten'],
            'Technik': ['wlan', 'app', 'klimaanlage', 'defekt', 'störung', 'kaputt'],
            'Sauberkeit': ['dreckig', 'sauber', 'toilette', 'müll', 'schmutz'],
            'Komfort': ['sitz', 'platz', 'überfüllt', 'bequem', 'eng', 'komfortabel'],
            'Pünktlichkeit': ['pünktlich', 'rechtzeitig', 'fahrplan', 'zeit']
        }

        topic_counts = {}
        for topic, keywords in topic_keywords.items():
            count = sum(1 for text in texts
                       if any(keyword in text.lower() for keyword in keywords))
            topic_counts[topic] = count

        return topic_counts

    def create_wordcloud(self):
        """Erstellt und zeigt eine WordCloud der häufigsten Wörter"""
        if self.analyzed_data is None:
            print("Keine Daten für WordCloud verfügbar. Führen Sie zuerst die Analyse durch.")
            return

        all_text = " ".join(self.analyzed_data['text'].tolist())
        # Import nltk here
        import nltk
        nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words('german')) # Annahme: Deutsche Stop-Wörter

        # Erstellen der WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=stop_words, min_font_size=10).generate(all_text)

        # Anzeigen der WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        print("✅ WordCloud erstellt")


    def analyze_all_posts(self):
        """Analysiert alle Posts und erstellt DataFrame"""

        if not self.posts:
            self.generate_realistic_posts()

        analyzed_data = []

        for post in self.posts:
            sentiment, score = self.analyze_sentiment_german(post['text'])

            analyzed_data.append({
                'text': post['text'],
                'platform': post['platform'],
                'date': post['date'],
                'sentiment': sentiment,
                'sentiment_score': score,
                'expected_sentiment': post['expected_sentiment'],
                'word_count': len(post['text'].split())
            })

        self.analyzed_data = pd.DataFrame(analyzed_data)

        # Topic-Extraktion
        all_texts = [post['text'] for post in self.posts]
        topics = self.extract_topics(all_texts)

        print("✅ Sentiment-Analyse abgeschlossen")
        print(f"Analysierte Posts: {len(self.analyzed_data)}")

        return self.analyzed_data, topics

    def create_comprehensive_analysis(self):
        """Erstellt umfassende Visualisierungen"""

        if self.analyzed_data is None:
            self.analyze_all_posts()

        # Setup für multiple Plots
        fig = plt.figure(figsize=(20, 24))

        # 1. Sentiment-Verteilung
        plt.subplot(4, 2, 1)
        sentiment_counts = self.analyzed_data['sentiment'].value_counts()
        colors = ['#ff4444', '#ffaa00', '#44aa44']
        wedges, texts, autotexts = plt.pie(sentiment_counts.values,
                                          labels=sentiment_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        plt.title('Sentiment-Verteilung: Deutsche Bahn Posts', fontsize=14, fontweight='bold')

        # 2. Sentiment nach Plattform
        plt.subplot(4, 2, 2)
        platform_sentiment = pd.crosstab(self.analyzed_data['platform'],
                                       self.analyzed_data['sentiment'])
        platform_sentiment.plot(kind='bar', stacked=True,
                               color=['#ff4444', '#ffaa00', '#44aa44'])
        plt.title('Sentiment nach Social Media Plattform', fontsize=14, fontweight='bold')
        plt.xlabel('Plattform')
        plt.ylabel('Anzahl Posts')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')

        # 3. Zeitlicher Verlauf
        plt.subplot(4, 2, 3)
        self.analyzed_data['date_only'] = self.analyzed_data['date'].dt.date
        daily_sentiment = self.analyzed_data.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0)

        if len(daily_sentiment) > 1:
            daily_sentiment.plot(kind='line', marker='o', linewidth=2,
                               color=['#ff4444', '#ffaa00', '#44aa44'])
            plt.title('Sentiment-Verlauf über Zeit', fontsize=14, fontweight='bold')
            plt.xlabel('Datum')
            plt.ylabel('Anzahl Posts')
            plt.xticks(rotation=45)
            plt.legend(title='Sentiment')

        # 4. Sentiment Score Verteilung
        plt.subplot(4, 2, 4)
        self.analyzed_data['sentiment_score'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Neutral (0)')
        plt.title('Verteilung der Sentiment-Scores', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Häufigkeit')
        plt.legend()

        # 5. Wort-Anzahl vs. Sentiment
        plt.subplot(4, 2, 5)
        for sentiment in ['negativ', 'neutral', 'positiv']:
            data = self.analyzed_data[self.analyzed_data['sentiment'] == sentiment]
            plt.scatter(data['word_count'], data['sentiment_score'],
                       alpha=0.6, label=sentiment, s=50)
        plt.title('Wortanzahl vs. Sentiment-Intensität', fontsize=14, fontweight='bold')
        plt.xlabel('Anzahl Wörter')
        plt.ylabel('Sentiment Score')
        plt.legend()

        # 6. Top Themen
        plt.subplot(4, 2, 6)
        all_texts = self.analyzed_data['text'].tolist()
        topics = self.extract_topics(all_texts)
        topic_df = pd.DataFrame(list(topics.items()), columns=['Thema', 'Häufigkeit'])
        topic_df = topic_df.sort_values('Häufigkeit', ascending=True)

        plt.barh(topic_df['Thema'], topic_df['Häufigkeit'], color='lightcoral')
        plt.title('Häufigste Themen in DB-Posts', fontsize=14, fontweight='bold')
        plt.xlabel('Anzahl Erwähnungen')

        # 7. Sentiment-Accuracy (Vergleich mit erwarteten Werten)
        plt.subplot(4, 2, 7)
        accuracy_data = pd.crosstab(self.analyzed_data['expected_sentiment'],
                                  self.analyzed_data['sentiment'], normalize='index') * 100

        sns.heatmap(accuracy_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   cbar_kws={'label': 'Prozent'})
        plt.title('Sentiment-Analyse Genauigkeit', fontsize=14, fontweight='bold')
        plt.xlabel('Erkanntes Sentiment')
        plt.ylabel('Erwartetes Sentiment')

        # 8. Durchschnittlicher Sentiment-Score nach Plattform
        plt.subplot(4, 2, 8)
        platform_avg = self.analyzed_data.groupby('platform')['sentiment_score'].mean()
        colors_platform = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        bars = plt.bar(platform_avg.index, platform_avg.values, color=colors_platform)
        plt.title('Durchschnittlicher Sentiment-Score nach Plattform', fontsize=14, fontweight='bold')
        plt.xlabel('Plattform')
        plt.ylabel('Avg. Sentiment Score')
        plt.xticks(rotation=45)

        # Werte auf Balken anzeigen
        for bar, value in zip(bars, platform_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # Zusätzliche WordCloud
        self.create_wordcloud()

        print("📊 Umfassende Analyse erstellt!")
        return self.analyzed_data

    def predict_sentiment_trends(self, days_ahead=7):
        """KI-Vorhersage für Sentiment-Trends"""

        if self.analyzed_data is None:
            self.analyze_all_posts()

        # Tägliche Sentiment-Durchschnitte berechnen
        daily_data = self.analyzed_data.groupby('date_only').agg({
            'sentiment_score': 'mean'
        }).reset_index()

        daily_data = daily_data.sort_values('date_only')

        # Konvertiere 'date_only' in datetime Objekte
        daily_data['date_only'] = pd.to_datetime(daily_data['date_only'])

        # Numerische Werte für Regression
        daily_data['days_since_start'] = (daily_data['date_only'] - daily_data['date_only'].min()).dt.days

        # Polynomial Regression für Trend-Vorhersage
        X = daily_data['days_since_start'].values.reshape(-1, 1)
        y = daily_data['sentiment_score'].values

        # Polynomial Features (Grad 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)

        # Model trainieren
        model = LinearRegression()
        model.fit(X_poly, y)

        # R² Score berechnen
        y_pred_current = model.predict(X_poly)
        r2 = r2_score(y, y_pred_current)

        # Vorhersage für nächste Tage
        future_days = np.arange(X[-1][0] + 1, X[-1][0] + days_ahead + 1).reshape(-1, 1)
        future_X_poly = poly_features.transform(future_days)
        future_predictions = model.predict(future_X_poly)

        # Visualisierung
        plt.figure(figsize=(14, 8))

        # Historische Daten
        plt.scatter(daily_data['days_since_start'], daily_data['sentiment_score'],
                   color='blue', alpha=0.7, s=50, label='Historische Daten')

        # Aktuelle Trend-Linie
        trend_line = model.predict(X_poly)
        plt.plot(daily_data['days_since_start'], trend_line,
                color='red', linewidth=2, label=f'Trend (R² = {r2:.3f})')

        # Vorhersagen
        future_days_flat = future_days.flatten()
        plt.plot(future_days_flat, future_predictions,
                color='orange', linewidth=2, linestyle='--', marker='o',
                label=f'KI-Vorhersage ({days_ahead} Tage)')

        # Styling
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Neutral')
        plt.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='Positiv-Schwelle')
        plt.axhline(y=-0.1, color='red', linestyle=':', alpha=0.5, label='Negativ-Schwelle')

        plt.title('KI-Vorhersage: Deutsche Bahn Sentiment-Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Tage seit Beginn der Analyse')
        plt.ylabel('Durchschnittlicher Sentiment-Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Interpretation ausgeben
        trend_direction = "steigend" if future_predictions[-1] > daily_data['sentiment_score'].iloc[-1] else "fallend"
        avg_future_sentiment = np.mean(future_predictions)

        print(f"\n🤖 KI-Trendvorhersage für die nächsten {days_ahead} Tage:")
        print(f"📈 Trend-Richtung: {trend_direction}")
        print(f"📊 Durchschnittliches Sentiment: {avg_future_sentiment:.3f}")
        print(f"🎯 Model-Genauigkeit (R²): {r2:.3f}")

        if avg_future_sentiment > 0.1:
            print("✅ Prognose: Positiver Sentiment-Trend erwartet")
        elif avg_future_sentiment < -0.1:
            print("⚠️ Prognose: Negativer Sentiment-Trend erwartet")
        else:
            print("📊 Prognose: Neutraler Sentiment-Trend erwartet")

        return future_predictions, r2

    def generate_summary_report(self):
        """Generiert einen zusammenfassenden Bericht"""

        if self.analyzed_data is None:
            self.analyze_all_posts()

        print("\n" + "="*60)
        print("📋 DEUTSCHE BAHN SENTIMENT-ANALYSE - ZUSAMMENFASSUNG")
        print("="*60)

        # Basis-Statistiken
        total_posts = len(self.analyzed_data)
        sentiment_dist = self.analyzed_data['sentiment'].value_counts()
        avg_sentiment = self.analyzed_data['sentiment_score'].mean()

        print(f"\n📊 ANALYSIERTE DATEN:")
        print(f"   • Gesamte Posts: {total_posts}")
        print(f"   • Analysezeitraum: 30 Tage")
        print(f"   • Durchschnittlicher Sentiment-Score: {avg_sentiment:.3f}")

        print(f"\n📈 SENTIMENT-VERTEILUNG:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total_posts) * 100
            print(f"   • {sentiment.capitalize()}: {count} Posts ({percentage:.1f}%)")

        # Plattform-Analyse
        platform_stats = self.analyzed_data.groupby('platform').agg({
            'sentiment_score': 'mean',
            'text': 'count'
        }).round(3)

        print(f"\n🌐 PLATTFORM-ANALYSE:")
        for platform, data in platform_stats.iterrows():
            print(f"   • {platform}: {data['text']} Posts, Ø Sentiment: {data['sentiment_score']:.3f}")

        # Themen-Analyse
        all_texts = self.analyzed_data['text'].tolist()
        topics = self.extract_topics(all_texts)
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"\n🏷️ TOP-THEMEN:")
        for topic, count in top_topics:
            print(f"   • {topic}: {count} Erwähnungen")

        print(f"\n🎯 ERKENNTNISSE:")
        if avg_sentiment < -0.1:
            print("   • Überwiegend negative Stimmung gegenüber der Deutschen Bahn")
        elif avg_sentiment > 0.1:
            print("   • Überwiegend positive Stimmung gegenüber der Deutschen Bahn")
        else:
            print("   • Neutrale bis gemischte Stimmung gegenüber der Deutschen Bahn")

        # Kritischste Plattform
        most_negative_platform = platform_stats['sentiment_score'].idxmin()
        most_positive_platform = platform_stats['sentiment_score'].idxmax()

        print(f"   • Kritischste Plattform: {most_negative_platform}")
        print(f"   • Positivste Plattform: {most_positive_platform}")
        print(f"   • Hauptkritikpunkte: Verspätungen, Ausfälle, Service")

        print("\n" + "="*60)

# Hauptausführung
def main():
    """Hauptfunktion - Führt die komplette Analyse durch"""

    print("🚀 Deutsche Bahn Sentiment-Analyse gestartet...")
    print("=" * 50)

    # Analyzer initialisieren
    analyzer = DeutscheBahnSentimentAnalyzer()

    # 1. Daten generieren und analysieren
    print("\n🔄 Schritt 1: Realistische Posts generieren...")
    analyzer.generate_realistic_posts()

    print("\n🔄 Schritt 2: Sentiment-Analyse durchführen...")
    data, topics = analyzer.analyze_all_posts()

    # 2. Visualisierungen erstellen
    print("\n🔄 Schritt 3: Visualisierungen erstellen...")
    analyzer.create_comprehensive_analysis()

    # 3. KI-Vorhersage
    print("\n🔄 Schritt 4: KI-Trendvorhersage...")
    predictions, accuracy = analyzer.predict_sentiment_trends(days_ahead=7)

    # 4. Zusammenfassender Bericht
    print("\n🔄 Schritt 5: Abschlussbericht generieren...")
    analyzer.generate_summary_report()

    print("\n✅ Analyse erfolgreich abgeschlossen!")
    print("💡 Alle Visualisierungen wurden angezeigt.")
    print("📊 Das Projekt demonstriert erfolgreich verschiedene KI-Technologien:")
    print("   • Text Mining & Sentiment Analysis")
    print("   • Topic Modeling & Data Visualization")
    print("   • Machine Learning Predictions")
    print("   • German Natural Language Processing")

#
