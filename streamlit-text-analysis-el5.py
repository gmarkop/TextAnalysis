import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from collections import Counter
import re
import numpy as np
import spacy
import matplotlib
import warnings
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for Greek support
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


# Cache expensive resource loading
@st.cache_resource
def load_spacy_model():
    """Load Greek spacy model once and cache it"""
    return spacy.load('el_core_news_sm')


@st.cache_data
def get_stopwords():
    """Cache Greek stopwords set"""
    return set(stopwords.words('greek'))


@st.cache_resource
def load_greek_sentiment_model():
    """Load Greek sentiment analysis model using XLM-RoBERTa"""
    try:
        # Using a multilingual model that works well with Greek
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,
                                     max_length=512, truncation=True)
        return sentiment_analyzer
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None


# Greek emotion lexicon (simplified)
GREEK_EMOTION_LEXICON = {
    'χαρά': {'joy': 1.0},
    'ευτυχία': {'joy': 1.0},
    'χαρούμενος': {'joy': 0.8},
    'χαρούμενη': {'joy': 0.8},
    'ευχαριστημένος': {'joy': 0.7, 'trust': 0.3},
    'ευχαριστημένη': {'joy': 0.7, 'trust': 0.3},
    'λύπη': {'sadness': 1.0},
    'λυπημένος': {'sadness': 0.8},
    'λυπημένη': {'sadness': 0.8},
    'θλίψη': {'sadness': 0.9},
    'στεναχώρια': {'sadness': 0.7},
    'θυμός': {'anger': 1.0},
    'θυμωμένος': {'anger': 0.8},
    'θυμωμένη': {'anger': 0.8},
    'εκνευρισμός': {'anger': 0.6},
    'οργή': {'anger': 0.9},
    'φόβος': {'fear': 1.0},
    'φοβισμένος': {'fear': 0.8},
    'φοβισμένη': {'fear': 0.8},
    'ανησυχία': {'fear': 0.6, 'anticipation': 0.3},
    'τρόμος': {'fear': 0.9},
    'έκπληξη': {'surprise': 1.0},
    'εκπληκτος': {'surprise': 0.8},
    'έκπληκτη': {'surprise': 0.8},
    'αηδία': {'disgust': 1.0},
    'απέχθεια': {'disgust': 0.9},
    'εμπιστοσύνη': {'trust': 1.0},
    'αξιοπιστία': {'trust': 0.8},
    'πίστη': {'trust': 0.7},
    'προσμονή': {'anticipation': 1.0},
    'προσδοκία': {'anticipation': 0.8},
    'αναμονή': {'anticipation': 0.7},
    'αγάπη': {'joy': 0.6, 'trust': 0.4},
    'μίσος': {'anger': 0.7, 'disgust': 0.3},
    'ελπίδα': {'anticipation': 0.6, 'joy': 0.4},
    'απελπισία': {'sadness': 0.6, 'fear': 0.4},
    'ενθουσιασμός': {'joy': 0.5, 'anticipation': 0.5},
    'απογοήτευση': {'sadness': 0.6, 'anger': 0.4},
}


class TextAnalyzer:
    def __init__(self, text):
        """
        Initialize the TextAnalyzer with Greek text.

        Args:
            text (str): The full text to be analyzed
        """
        # Use cached spacy model
        self.nlp = load_spacy_model()

        # Clean and store text
        self.original_text = self.clean_text_for_display(text)

        # Process with spacy once and cache the doc
        self.doc = self.nlp(self.original_text)

        # Preprocess text
        self.processed_text = self.preprocess_text(self.original_text)

        # Tokenize once and store
        self.tokens = word_tokenize(self.processed_text)
        self.sentences = sent_tokenize(self.original_text)

        # Lazy initialization
        self._cleaned_tokens = None
        self._pos_counts = None
        self._sentiment_analyzer = None

    @property
    def sentiment_analyzer(self):
        """Lazy loading of sentiment analyzer"""
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = load_greek_sentiment_model()
        return self._sentiment_analyzer

    @staticmethod
    def clean_text_for_display(text):
        """Clean text of problematic characters"""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        return text

    @staticmethod
    def preprocess_text(text):
        """Preprocess Greek text: lowercase, remove non-Greek chars, normalize whitespace"""
        text = text.lower()
        text = re.sub(r'[^α-ωά-ώϊϋΐΰ\s]', '', text)
        text = ' '.join(text.split())
        return text

    def get_cleaned_tokens(self):
        """Get tokens with stopwords removed (cached)"""
        if self._cleaned_tokens is None:
            stop_words = get_stopwords()
            self._cleaned_tokens = [token for token in self.tokens if token not in stop_words]
        return self._cleaned_tokens

    def get_word_frequencies(self, top_n=20):
        """Get word frequency distribution"""
        cleaned_tokens = self.get_cleaned_tokens()
        freq_dist = FreqDist(cleaned_tokens)

        freq_df = pd.DataFrame(
            freq_dist.most_common(top_n),
            columns=['Word', 'Frequency']
        )

        return freq_df

    def get_pos_counts(self):
        """Get POS counts (cached)"""
        if self._pos_counts is None:
            self._pos_counts = Counter([token.pos_ for token in self.doc])
        return self._pos_counts

    def pos_analysis(self):
        """Analyze parts of speech"""
        pos_counts = self.get_pos_counts()

        pos_df = pd.DataFrame(
            list(pos_counts.items()),
            columns=['POS Tag', 'Frequency']
        ).sort_values('Frequency', ascending=False)

        pos_df.index = range(1, len(pos_df) + 1)

        return pos_df

    def basic_statistics(self):
        """Calculate basic text statistics"""
        token_count = len(self.tokens)
        sentence_count = len(self.sentences)

        return {
            'Total Words': token_count,
            'Total Sentences': sentence_count,
            'Average Word Length': np.mean([len(word) for word in self.tokens]) if self.tokens else 0,
            'Average Sentence Length': token_count / sentence_count if sentence_count > 0 else 0
        }

    def sentiment_analysis(self):
        """
        Analyze sentiment using Greek-capable transformer model.

        Returns:
            dict: Dictionary containing sentiment polarity and subjectivity
        """
        if self.sentiment_analyzer is None:
            return {'Overall Polarity': 0.0, 'Overall Subjectivity': 0.5, 'error': True}

        try:
            # Truncate text if too long
            text_to_analyze = self.original_text[:512]
            result = self.sentiment_analyzer(text_to_analyze)[0]

            # Map labels to polarity
            label_map = {
                'positive': 0.5,
                'Positive': 0.5,
                'POSITIVE': 0.5,
                'neutral': 0.0,
                'Neutral': 0.0,
                'NEUTRAL': 0.0,
                'negative': -0.5,
                'Negative': -0.5,
                'NEGATIVE': -0.5
            }

            polarity = label_map.get(result['label'], 0.0)
            confidence = result['score']

            # Adjust polarity based on confidence
            polarity = polarity * confidence

            return {
                'Overall Polarity': polarity,
                'Overall Subjectivity': confidence,
                'label': result['label'],
                'confidence': confidence
            }
        except Exception as e:
            return {'Overall Polarity': 0.0, 'Overall Subjectivity': 0.5, 'error': str(e)}

    def sentence_sentiment_analysis(self):
        """
        Analyze sentiment for each sentence.

        Returns:
            pd.DataFrame: DataFrame containing sentences and their sentiment scores
        """
        if self.sentiment_analyzer is None:
            return pd.DataFrame()

        data = {'Sentence': [], 'Polarity': [], 'Subjectivity': [], 'Sentiment': []}

        for sentence in self.sentences:
            try:
                # Truncate sentence if too long
                sentence_text = sentence[:512]
                result = self.sentiment_analyzer(sentence_text)[0]

                # Map labels to polarity
                label_map = {
                    'positive': 0.5, 'Positive': 0.5, 'POSITIVE': 0.5,
                    'neutral': 0.0, 'Neutral': 0.0, 'NEUTRAL': 0.0,
                    'negative': -0.5, 'Negative': -0.5, 'NEGATIVE': -0.5
                }

                polarity = label_map.get(result['label'], 0.0) * result['score']
                confidence = result['score']

                # Determine sentiment label
                if polarity > 0.1:
                    sentiment = 'Positive'
                elif polarity < -0.1:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'

                data['Sentence'].append(sentence)
                data['Polarity'].append(polarity)
                data['Subjectivity'].append(confidence)
                data['Sentiment'].append(sentiment)

            except Exception:
                data['Sentence'].append(sentence)
                data['Polarity'].append(0.0)
                data['Subjectivity'].append(0.5)
                data['Sentiment'].append('Neutral')

        return pd.DataFrame(data)

    def emotion_analysis_from_lexicon(self, text):
        """
        Analyze emotions using Greek lexicon.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Emotion scores
        """
        tokens = word_tokenize(text.lower())
        emotion_scores = {
            'joy': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0,
            'surprise': 0.0, 'disgust': 0.0, 'trust': 0.0, 'anticipation': 0.0
        }

        total_emotional_words = 0

        for token in tokens:
            if token in GREEK_EMOTION_LEXICON:
                emotions = GREEK_EMOTION_LEXICON[token]
                for emotion, score in emotions.items():
                    emotion_scores[emotion] += score
                total_emotional_words += 1

        # Normalize scores
        if total_emotional_words > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total_emotional_words

        return emotion_scores

    def emotion_analysis(self):
        """
        Analyze emotions in the text.

        Returns:
            dict: Dictionary containing emotion scores
        """
        return self.emotion_analysis_from_lexicon(self.original_text)

    def sentence_emotion_analysis(self):
        """
        Analyze emotions for each sentence.

        Returns:
            pd.DataFrame: DataFrame containing sentences and their emotion scores
        """
        sentence_emotions = []

        for sentence in self.sentences:
            emotions = self.emotion_analysis_from_lexicon(sentence)

            emotion_dict = {
                'Sentence': sentence,
                'Joy': emotions['joy'],
                'Sadness': emotions['sadness'],
                'Anger': emotions['anger'],
                'Fear': emotions['fear'],
                'Surprise': emotions['surprise'],
                'Disgust': emotions['disgust'],
                'Trust': emotions['trust'],
                'Anticipation': emotions['anticipation']
            }

            # Find dominant emotion
            emotion_scores = {k: v for k, v in emotion_dict.items() if k != 'Sentence'}
            if any(emotion_scores.values()):
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                emotion_dict['Dominant Emotion'] = dominant_emotion
            else:
                emotion_dict['Dominant Emotion'] = 'Neutral'

            sentence_emotions.append(emotion_dict)

        return pd.DataFrame(sentence_emotions)

    def topic_modeling_lda(self, n_topics=5, n_top_words=10, method='lda'):
        """
        Perform topic modeling using LDA or NMF.

        Args:
            n_topics (int): Number of topics to extract
            n_top_words (int): Number of top words per topic
            method (str): 'lda' or 'nmf'

        Returns:
            dict: Dictionary containing topic information
        """
        if len(self.sentences) < 3:
            return None

        documents = self.sentences
        stop_words = list(get_stopwords())

        # Create document-term matrix
        if method == 'lda':
            vectorizer = CountVectorizer(
                max_df=0.85,
                min_df=2,
                stop_words=stop_words,
                max_features=1000,
                token_pattern=r'[α-ωά-ώϊϋΐΰ]+'
            )
        else:  # nmf
            vectorizer = TfidfVectorizer(
                max_df=0.85,
                min_df=2,
                stop_words=stop_words,
                max_features=1000,
                token_pattern=r'[α-ωά-ώϊϋΐΰ]+'
            )

        try:
            doc_term_matrix = vectorizer.fit_transform(documents)

            # Fit the model
            if method == 'lda':
                model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=20,
                    learning_method='online'
                )
            else:  # nmf
                model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=200
                )

            doc_topic_dist = model.fit_transform(doc_term_matrix)

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            # Extract topics
            topics = {}
            for topic_idx, topic in enumerate(model.components_):
                top_indices = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                top_weights = [topic[i] for i in top_indices]
                topics[f'Topic {topic_idx + 1}'] = {
                    'words': top_words,
                    'weights': top_weights
                }

            # Get dominant topic for each document
            dominant_topics = []
            for i, doc_dist in enumerate(doc_topic_dist):
                dominant_topic = doc_dist.argmax()
                dominant_topics.append({
                    'Sentence': documents[i][:100] + '...' if len(documents[i]) > 100 else documents[i],
                    'Dominant Topic': f'Topic {dominant_topic + 1}',
                    'Topic Weight': doc_topic_dist[dominant_topic]
                })

            return {
                'topics': topics,
                'doc_topic_dist': doc_topic_dist,
                'dominant_topics': pd.DataFrame(dominant_topics),
                'model': model,
                'vectorizer': vectorizer
            }

        except Exception as e:
            return None


def create_safe_plot(figsize=(10, 6)):
    """Create a matplotlib figure with safe settings"""
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


@st.cache_data
def analyze_text(text):
    """Cache the text analyzer initialization"""
    return TextAnalyzer(text)


def main():
    st.title('Εργαλείο Ανάλυσης Κειμένου')  # Text Analysis Tool in Greek

    with st.sidebar:
        st.header("Επιλογές Ανάλυσης")  # Analysis Options
        analysis_tab = st.radio(
            "Επιλέξτε Τύπο Ανάλυσης",  # Select Analysis Type
            ["Βασική Ανάλυση", "Ανάλυση Συναισθήματος", "Ανάλυση Συναισθημάτων", "Θεματική Μοντελοποίηση"]
        )

    # File upload
    uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο κειμένου", type=['txt'])

    if uploaded_file is not None:
        try:
            # Read the file
            text = uploaded_file.getvalue().decode("utf-8")
            with st.expander("Προβολή Περιεχομένου Κειμένου", expanded=False):
                st.text_area('Περιεχόμενο Κειμένου', text, height=200)

            # Create analyzer with caching
            analyzer = analyze_text(text)

            # Display basic statistics
            stats = analyzer.basic_statistics()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Σύνολο Λέξεων", f"{stats['Total Words']:.0f}")
            with col2:
                st.metric("Σύνολο Προτάσεων", f"{stats['Total Sentences']:.0f}")
            with col3:
                st.metric("Μέσο Μήκος Λέξης", f"{stats['Average Word Length']:.2f}")
            with col4:
                st.metric("Μέσο Μήκος Πρότασης", f"{stats['Average Sentence Length']:.2f}")

            if analysis_tab == "Βασική Ανάλυση":
                # Word Frequency
                st.header('Συχνότητες Λέξεων')
                col1, col2 = st.columns([1, 2])

                with col1:
                    top_n = st.slider('Επιλέξτε αριθμό κορυφαίων λέξεων', 5, 50, 20)
                    freq_df = analyzer.get_word_frequencies(top_n)
                    freq_df.index = freq_df.index + 1
                    st.dataframe(freq_df, use_container_width=True)

                with col2:
                    try:
                        fig, ax = create_safe_plot()
                        freq_data = freq_df.sort_values('Frequency', ascending=True).tail(15)
                        sns.barplot(x='Frequency', y='Word', data=freq_data, ax=ax)
                        ax.set_title('Συχνότητες Λέξεων')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Σφάλμα δημιουργίας γραφήματος: {str(e)}")

                # POS Tag Frequencies
                st.header('Ανάλυση Μερών του Λόγου')
                col1, col2 = st.columns([1, 2])

                with col1:
                    pos_df = analyzer.pos_analysis()
                    st.dataframe(pos_df, use_container_width=True)

                with col2:
                    try:
                        fig, ax = create_safe_plot()
                        pos_data = pos_df.sort_values('Frequency', ascending=True).tail(10)
                        sns.barplot(x='Frequency', y='POS Tag', data=pos_data, ax=ax)
                        ax.set_title('Κορυφαίες 10 Συχνότητες Μερών του Λόγου')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Σφάλμα δημιουργίας γραφήματος: {str(e)}")

                # Tagged Text
                with st.expander("Προβολή Σημειωμένου Κειμένου", expanded=False):
                    try:
                        pos_text = ""
                        for sent in analyzer.doc.sents:
                            for token in sent:
                                pos_text += f"{token.text}[{token.pos_}] "
                            pos_text += "\n\n"
                        st.text_area('Σημειωμένο Κείμενο', pos_text, height=300)
                    except Exception as e:
                        st.error(f"Σφάλμα επεξεργασίας κειμένου: {str(e)}")

            elif analysis_tab == "Ανάλυση Συναισθήματος":
                st.header('Ανάλυση Συναισθήματος')

                # Overall Sentiment
                sentiment = analyzer.sentiment_analysis()

                if 'error' not in sentiment or not sentiment['error']:
                    polarity = sentiment['Overall Polarity']
                    subjectivity = sentiment['Overall Subjectivity']

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Πολικότητα Συναισθήματος")
                        try:
                            fig, ax = create_safe_plot()
                            ax.set_xlim(-1, 1)
                            ax.set_ylim(0, 0.1)
                            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                            ax.axvspan(-1, -0.1, color='red', alpha=0.2)
                            ax.axvspan(-0.1, 0.1, color='gray', alpha=0.2)
                            ax.axvspan(0.1, 1, color='green', alpha=0.2)
                            ax.scatter(polarity, 0.05, s=300, color='blue', zorder=5)

                            ax.set_yticks([])
                            ax.set_ylabel('')
                            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                            ax.set_xticklabels(['Αρνητικό', 'Κάπως\nΑρνητικό', 'Ουδέτερο', 'Κάπως\nΘετικό', 'Θετικό'])
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Σφάλμα δημιουργίας γραφήματος: {str(e)}")

                        st.metric("Βαθμολογία Πολικότητας", f"{polarity:.3f}",
                                  delta=None if -0.1 <= polarity <= 0.1 else ("Θετικό" if polarity > 0 else "Αρνητικό"))

                    with col2:
                        st.subheader("Υποκειμενικότητα")
                        try:
                            fig, ax = create_safe_plot()
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 0.1)
                            ax.axvspan(0, 0.3, color='lightblue', alpha=0.3)
                            ax.axvspan(0.3, 0.7, color='lightgreen', alpha=0.3)
                            ax.axvspan(0.7, 1, color='lightyellow', alpha=0.3)
                            ax.scatter(subjectivity, 0.05, s=300, color='blue', zorder=5)

                            ax.set_yticks([])
                            ax.set_ylabel('')
                            ax.set_xticks([0, 0.3, 0.7, 1])
                            ax.set_xticklabels(['Αντικειμενικό', 'Κάπως\nΑντικειμενικό', 'Κάπως\nΥποκειμενικό', 'Υποκειμενικό'])
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Σφάλμα δημιουργίας γραφήματος: {str(e)}")

                        st.metric("Βαθμολογία Υποκειμενικότητας", f"{subjectivity:.3f}",
                                  delta="Πραγματικό" if subjectivity < 0.5 else "Γνωμικό")

                    # Sentence Level Sentiment
                    st.subheader('Ανάλυση Συναισθήματος σε Επίπεδο Πρότασης')
                    sent_df = analyzer.sentence_sentiment_analysis()

                    if not sent_df.empty:
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            try:
                                fig, ax = create_safe_plot()
                                sentiment_counts = sent_df['Sentiment'].value_counts()
                                colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                                bars = ax.bar(sentiment_counts.index, sentiment_counts.values,
                                             color=[colors.get(x, 'blue') for x in sentiment_counts.index])
                                ax.set_title('Κατανομή Συναισθήματος Προτάσεων')
                                ax.set_ylabel('Αριθμός Προτάσεων')

                                for i, v in enumerate(sentiment_counts.values):
                                    ax.text(i, v + 0.1, str(v), ha='center')

                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:
                                st.error(f"Σφάλμα δημιουργίας γραφήματος: {str(e)}")

                        with col2:
                            sentiment_counts_dict = sent_df['Sentiment'].value_counts().to_dict()
                            positive_count = sentiment_counts_dict.get('Positive', 0)
                            negative_count = sentiment_counts_dict.get('Negative', 0)
                            neutral_count = sentiment_counts_dict.get('Neutral', 0)
                            total_sentences = len(sent_df)

                            if total_sentences > 0:
                                st.metric("Θετικές Προτάσεις", positive_count, f"{positive_count/total_sentences*100:.1f}%")
                                st.metric("Ουδέτερες Προτάσεις", neutral_count, f"{neutral_count/total_sentences*100:.1f}%")
                                st.metric("Αρνητικές Προτάσεις", negative_count, f"{negative_count/total_sentences*100:.1f}%")

                        with st.expander("Προβολή Λεπτομερούς Ανάλυσης Προτάσεων", expanded=False):
                            st.dataframe(sent_df, use_container_width=True)
                else:
                    st.error("Σφάλμα φόρτωσης μοντέλου ανάλυσης συναισθήματος. Παρακαλώ ελέγξτε τη σύνδεσή σας στο διαδίκτυο.")

            elif analysis_tab == "Ανάλυση Συναισθημάτων":
                st.header('Ανάλυση Συναισθημάτων')

                emotions = analyzer.emotion_analysis()

                st.subheader("Συνολική Κατανομή Συναισθημάτων")

                emotion_df = pd.DataFrame({
                    'Συναίσθημα': ['Χαρά', 'Λύπη', 'Θυμός', 'Φόβος', 'Έκπληξη', 'Αηδία', 'Εμπιστοσύνη', 'Προσμονή'],
                    'Score': [emotions['joy'], emotions['sadness'], emotions['anger'], emotions['fear'],
                             emotions['surprise'], emotions['disgust'], emotions['trust'], emotions['anticipation']]
                }).sort_values('Score', ascending=False)

                emotion_colors = {
                    'Χαρά': '#FFCC00',
                    'Εμπιστοσύνη': '#4CAF50',
                    'Προσμονή': '#FF9800',
                    'Έκπληξη': '#8E44AD',
                    'Λύπη': '#3498DB',
                    'Φόβος': '#607D8B',
                    'Θυμός': '#F44336',
                    'Αηδία': '#795548'
                }

                colors = [emotion_colors.get(emotion, '#CCCCCC') for emotion in emotion_df['Συναίσθημα']]

                try:
                    fig, ax = create_safe_plot()
                    bars = ax.barh(emotion_df['Συναίσθημα'], emotion_df['Score'], color=colors)
                    ax.set_xlabel('Βαθμολογία')
                    ax.set_title('Ανάλυση Συναισθημάτων')

                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        if width > 0:
                            label_x_pos = width if width > 0.02 else 0.02
                            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                                    va='center', ha='left' if width <= 0.02 else 'right',
                                    color='black' if width <= 0.02 else 'white')

                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Σφάλμα δημιουργίας γραφήματος: {str(e)}")

                if any(emotions.values()):
                    dominant_emotion = max(emotions, key=emotions.get)
                    dominant_score = emotions[dominant_emotion]

                    emotion_greek_map = {
                        'joy': 'Χαρά', 'sadness': 'Λύπη', 'anger': 'Θυμός', 'fear': 'Φόβος',
                        'surprise': 'Έκπληξη', 'disgust': 'Αηδία', 'trust': 'Εμπιστοσύνη', 'anticipation': 'Προσμονή'
                    }

                    st.subheader("Κυρίαρχο Συναίσθημα")
                    st.markdown(f"Το κυρίαρχο συναίσθημα σε αυτό το κείμενο είναι: **{emotion_greek_map.get(dominant_emotion, dominant_emotion)}** (Βαθμολογία: {dominant_score:.3f})")
                else:
                    st.info("Δεν εντοπίστηκαν σημαντικά συναισθήματα στο κείμενο.")

                # Emotion radar chart
                st.subheader("Προφίλ Συναισθημάτων")

                try:
                    categories = ['Χαρά', 'Λύπη', 'Θυμός', 'Φόβος', 'Έκπληξη', 'Αηδία', 'Εμπιστοσύνη', 'Προσμονή']
                    values = [emotions['joy'], emotions['sadness'], emotions['anger'], emotions['fear'],
                             emotions['surprise'], emotions['disgust'], emotions['trust'], emotions['anticipation']]

                    if any(values):
                        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                        values = values + [values[0]]
                        angles = angles + [angles[0]]
                        categories = categories + [categories[0]]

                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111, polar=True)

                        ax.plot(angles, values, 'o-', linewidth=2)
                        ax.fill(angles, values, alpha=0.25)

                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(categories[:-1])

                        max_value = max(values)
                        ax.set_ylim(0, max_value * 1.1 if max_value > 0 else 0.1)
                        ax.grid(True)

                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Δεν εντοπίστηκαν σημαντικά συναισθήματα στο κείμενο.")
                except Exception as e:
                    st.error(f"Σφάλμα δημιουργίας γραφήματος: {str(e)}")

                # Sentence-level emotion analysis
                st.subheader("Ανάλυση Συναισθημάτων σε Επίπεδο Πρότασης")
                try:
                    sentence_emotions = analyzer.sentence_emotion_analysis()

                    emotion_columns = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Trust', 'Anticipation']

                    display_limit = min(20, len(sentence_emotions))
                    heatmap_data = sentence_emotions.iloc[:display_limit][emotion_columns]

                    sentence_labels = [f"Π{i+1}: {s[:30]}..." if len(s) > 30 else f"Π{i+1}: {s}"
                            for i, s in enumerate(sentence_emotions['Sentence'].iloc[:display_limit])]

                    if not heatmap_data.empty and heatmap_data.sum().sum() > 0:
                        fig, ax = plt.subplots(figsize=(12, max(6, display_limit * 0.4)))

                        # Rename columns to Greek for heatmap
                        heatmap_data_greek = heatmap_data.copy()
                        heatmap_data_greek.columns = ['Χαρά', 'Λύπη', 'Θυμός', 'Φόβος', 'Έκπληξη', 'Αηδία', 'Εμπιστοσύνη', 'Προσμονή']

                        sns.heatmap(heatmap_data_greek, annot=True, cmap="YlGnBu",
                                   yticklabels=sentence_labels, fmt='.2f', ax=ax)
                        ax.set_title('Κατανομή Συναισθημάτων σε Προτάσεις')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Δεν εντοπίστηκαν σημαντικά συναισθήματα στις επιμέρους προτάσεις.")

                    with st.expander("Προβολή Λεπτομερών Συναισθημάτων Προτάσεων", expanded=False):
                        cols = ['Sentence', 'Dominant Emotion'] + emotion_columns
                        st.dataframe(sentence_emotions[cols], use_container_width=True)

                except Exception as e:
                    st.error(f"Σφάλμα στην ανάλυση συναισθημάτων σε επίπεδο πρότασης: {str(e)}")

            elif analysis_tab == "Θεματική Μοντελοποίηση":
                st.header('Ανάλυση Θεματικής Μοντελοποίησης')

                st.markdown("""
                Η θεματική μοντελοποίηση βοηθά στην αναγνώριση των κύριων θεμάτων στο κείμενό σας.
                Αυτή η ανάλυση χρησιμοποιεί **Latent Dirichlet Allocation (LDA)** ή **Non-negative Matrix Factorization (NMF)**
                για την ανακάλυψη κρυφών θεμάτων βάσει προτύπων λέξεων.
                """)

                if len(analyzer.sentences) < 3:
                    st.warning("Η θεματική μοντελοποίηση απαιτεί τουλάχιστον 3 προτάσεις. Το κείμενό σας είναι πολύ σύντομο για ουσιαστική ανάλυση θεμάτων.")
                else:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        method = st.selectbox(
                            "Επιλέξτε Μέθοδο",
                            ["lda", "nmf"],
                            format_func=lambda x: "LDA (Latent Dirichlet Allocation)" if x == "lda" else "NMF (Non-negative Matrix Factorization)"
                        )

                    with col2:
                        n_topics = st.slider("Αριθμός Θεμάτων", min_value=2, max_value=10, value=5)

                    with col3:
                        n_words = st.slider("Λέξεις ανά Θέμα", min_value=5, max_value=20, value=10)

                    try:
                        with st.spinner('Εξαγωγή θεμάτων...'):
                            topic_results = analyzer.topic_modeling_lda(
                                n_topics=n_topics,
                                n_top_words=n_words,
                                method=method
                            )

                        if topic_results is None:
                            st.error("Δεν ήταν δυνατή η εκτέλεση θεματικής μοντελοποίησης. Βεβαιωθείτε ότι το κείμενό σας έχει αρκετό περιεχόμενο και ποικιλία.")
                        else:
                            topics = topic_results['topics']
                            doc_topic_dist = topic_results['doc_topic_dist']
                            dominant_topics_df = topic_results['dominant_topics']

                            # Display topics
                            st.subheader("Ανακαλυφθέντα Θέματα")

                            cols_per_row = 2
                            topic_items = list(topics.items())

                            for i in range(0, len(topic_items), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, (topic_name, topic_data) in enumerate(topic_items[i:i+cols_per_row]):
                                    with cols[j]:
                                        st.markdown(f"### {topic_name}")
                                        words_str = ", ".join(topic_data['words'][:8])
                                        st.info(f"**Κορυφαίες λέξεις:** {words_str}")

                            # Visualize topics
                            st.subheader("Κατανομές Λέξεων Θεμάτων")

                            for topic_name, topic_data in topics.items():
                                try:
                                    fig, ax = create_safe_plot(figsize=(10, 4))

                                    words = topic_data['words'][:n_words]
                                    weights = topic_data['weights'][:n_words]

                                    weights = np.array(weights)
                                    weights = weights / weights.sum()

                                    colors_palette = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))

                                    bars = ax.barh(range(len(words)), weights, color=colors_palette)
                                    ax.set_yticks(range(len(words)))
                                    ax.set_yticklabels(words)
                                    ax.set_xlabel('Βάρος')
                                    ax.set_title(f'{topic_name} - Κορυφαίες Λέξεις')
                                    ax.invert_yaxis()

                                    for i, (bar, weight) in enumerate(zip(bars, weights)):
                                        width = bar.get_width()
                                        ax.text(width, bar.get_y() + bar.get_height()/2,
                                               f' {weight:.3f}',
                                               va='center', ha='left', fontsize=9)

                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)

                                except Exception as e:
                                    st.error(f"Σφάλμα δημιουργίας οπτικοποίησης για {topic_name}: {str(e)}")

                            # Topic distribution
                            st.subheader("Κατανομή Θεμάτων")

                            try:
                                fig, ax = create_safe_plot(figsize=(12, 6))

                                display_limit = min(30, len(doc_topic_dist))
                                heatmap_data = doc_topic_dist[:display_limit]

                                im = ax.imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd')

                                ax.set_yticks(range(n_topics))
                                ax.set_yticklabels([f'Θέμα {i+1}' for i in range(n_topics)])
                                ax.set_xlabel('Δείκτης Πρότασης')
                                ax.set_ylabel('Θέματα')
                                ax.set_title('Κατανομή Θεμάτων σε Προτάσεις')

                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('Βάρος Θέματος')

                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                            except Exception as e:
                                st.error(f"Σφάλμα δημιουργίας οπτικοποίησης κατανομής θεμάτων: {str(e)}")

                            # Overall topic proportions
                            st.subheader("Συνολικές Αναλογίες Θεμάτων")

                            try:
                                avg_topic_weights = doc_topic_dist.mean(axis=0)

                                fig, ax = plt.subplots(figsize=(8, 8))

                                colors_pie = plt.cm.Set3(range(n_topics))
                                patches, texts, autotexts = ax.pie(
                                    avg_topic_weights,
                                    labels=[f'Θέμα {i+1}' for i in range(n_topics)],
                                    autopct='%1.1f%%',
                                    colors=colors_pie,
                                    startangle=90
                                )

                                for autotext in autotexts:
                                    autotext.set_color('white')
                                    autotext.set_fontweight('bold')

                                ax.axis('equal')
                                ax.set_title('Μέσες Αναλογίες Θεμάτων στο Κείμενο')
                                st.pyplot(fig)
                                plt.close(fig)

                            except Exception as e:
                                st.error(f"Σφάλμα δημιουργίας γραφήματος αναλογιών θεμάτων: {str(e)}")

                            # Sentence-topic assignments
                            st.subheader("Ανάθεση Προτάσεων σε Θέματα")

                            st.markdown("Κάθε πρότασηανατίθεται στο πιο κυρίαρχο θέμα της:")

                            topic_counts = dominant_topics_df['Dominant Topic'].value_counts()
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                st.markdown("**Κατανομή Θεμάτων:**")
                                for topic, count in topic_counts.items():
                                    percentage = (count / len(dominant_topics_df)) * 100
                                    st.metric(topic, count, f"{percentage:.1f}%")

                            with col2:
                                try:
                                    fig, ax = create_safe_plot()
                                    colors_bar = plt.cm.Set3(range(len(topic_counts)))
                                    bars = ax.bar(topic_counts.index, topic_counts.values, color=colors_bar)
                                    ax.set_xlabel('Θέμα')
                                    ax.set_ylabel('Αριθμός Προτάσεων')
                                    ax.set_title('Προτάσεις ανά Θέμα')

                                    for bar in bars:
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                               f'{int(height)}',
                                               ha='center', va='bottom')

                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)

                                except Exception as e:
                                    st.error(f"Σφάλμα δημιουργίας γραφήματος κατανομής θεμάτων: {str(e)}")

                            with st.expander("Προβολή Λεπτομερών Αναθέσεων Προτάσεων-Θεμάτων", expanded=False):
                                sorted_df = dominant_topics_df.sort_values('Topic Weight', ascending=False)
                                st.dataframe(sorted_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Σφάλμα εκτέλεσης θεματικής μοντελοποίησης: {str(e)}")
                        st.info("Δοκιμάστε να προσαρμόσετε τον αριθμό θεμάτων ή βεβαιωθείτε ότι το κείμενό σας έχει επαρκές περιεχόμενο.")

        except Exception as e:
            st.error(f"Παρουσιάστηκε σφάλμα κατά την επεξεργασία του αρχείου: {str(e)}")
            st.error("Παρακαλώ βεβαιωθείτε ότι το αρχείο είναι έγκυρο αρχείο κειμένου και δοκιμάστε ξανά.")

if __name__ == "__main__":
    main()
