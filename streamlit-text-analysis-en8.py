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
from textblob import TextBlob
from nrclex import NRCLex
import matplotlib
import warnings
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use a safe backend and font
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


# Cache expensive resource loading
@st.cache_resource
def load_spacy_model():
    """Load spacy model once and cache it"""
    return spacy.load('en_core_web_sm')


@st.cache_data
def get_stopwords():
    """Cache stopwords set"""
    return set(stopwords.words('english'))


class TextAnalyzer:
    def __init__(self, text):
        """
        Initialize the TextAnalyzer with a given text.

        Args:
            text (str): The full text to be analyzed
        """
        # Use cached spacy model
        self.nlp = load_spacy_model()

        # Clean text once
        self.original_text = self.clean_text_for_display(text)

        # Process with spacy once and cache the doc
        self.doc = self.nlp(self.original_text)

        # Preprocess text
        self.processed_text = self.preprocess_text(self.original_text)

        # Tokenize once and store
        self.tokens = word_tokenize(self.processed_text)
        self.sentences = sent_tokenize(self.original_text)

        # Create TextBlob once
        self.textblob = TextBlob(self.original_text)

        # Lazy initialization for emotion analyzer (only when needed)
        self._emotion_analyzer = None
        self._cleaned_tokens = None
        self._pos_counts = None

    @property
    def emotion_analyzer(self):
        """Lazy loading of emotion analyzer"""
        if self._emotion_analyzer is None:
            self._emotion_analyzer = NRCLex(self.original_text)
        return self._emotion_analyzer

    @staticmethod
    def clean_text_for_display(text):
        """Clean text of problematic characters that cause font rendering issues"""
        # Remove carriage returns and normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        return text

    @staticmethod
    def preprocess_text(text):
        """Preprocess text: lowercase, remove non-alpha, normalize whitespace"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
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
        Analyze the sentiment of the text using TextBlob.

        Returns:
            dict: Dictionary containing sentiment polarity and subjectivity
        """
        sentiment = self.textblob.sentiment
        return {
            'Overall Polarity': sentiment.polarity,
            'Overall Subjectivity': sentiment.subjectivity
        }

    def sentence_sentiment_analysis(self):
        """
        Analyze sentiment for each sentence in the text.

        Returns:
            pd.DataFrame: DataFrame containing sentences and their sentiment scores
        """
        # Vectorized sentiment extraction
        data = {
            'Sentence': [str(sentence) for sentence in self.textblob.sentences],
            'Polarity': [sentence.sentiment.polarity for sentence in self.textblob.sentences],
            'Subjectivity': [sentence.sentiment.subjectivity for sentence in self.textblob.sentences]
        }

        sentiment_df = pd.DataFrame(data)

        # Vectorized sentiment labeling
        sentiment_df['Sentiment'] = pd.cut(
            sentiment_df['Polarity'],
            bins=[-float('inf'), -0.1, 0.1, float('inf')],
            labels=['Negative', 'Neutral', 'Positive']
        )

        return sentiment_df

    def emotion_analysis(self):
        """
        Analyze emotions in the text using NRCLex.

        Returns:
            dict: Dictionary containing emotion scores
        """
        emotions = self.emotion_analyzer.affect_frequencies

        # Return core emotions in consistent order
        core_emotions = {
            'fear': emotions.get('fear', 0),
            'anger': emotions.get('anger', 0),
            'anticipation': emotions.get('anticipation', 0),
            'trust': emotions.get('trust', 0),
            'surprise': emotions.get('surprise', 0),
            'sadness': emotions.get('sadness', 0),
            'disgust': emotions.get('disgust', 0),
            'joy': emotions.get('joy', 0)
        }

        return core_emotions

    def sentence_emotion_analysis(self):
        """
        Analyze emotions for each sentence in the text.

        Returns:
            pd.DataFrame: DataFrame containing sentences and their emotion scores
        """
        sentence_emotions = []

        for sentence in self.sentences:
            # Analyze emotions per sentence
            emotion_analyzer = NRCLex(sentence)
            emotions = emotion_analyzer.affect_frequencies

            # Build emotion dictionary
            emotion_dict = {
                'Sentence': sentence,
                'Joy': emotions.get('joy', 0),
                'Sadness': emotions.get('sadness', 0),
                'Anger': emotions.get('anger', 0),
                'Fear': emotions.get('fear', 0),
                'Surprise': emotions.get('surprise', 0),
                'Disgust': emotions.get('disgust', 0),
                'Trust': emotions.get('trust', 0),
                'Anticipation': emotions.get('anticipation', 0)
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
        # Check if we have enough sentences
        if len(self.sentences) < 3:
            return None

        # Prepare documents (use sentences as documents)
        documents = self.sentences

        # Create document-term matrix
        if method == 'lda':
            vectorizer = CountVectorizer(
                max_df=0.85,
                min_df=2,
                stop_words='english',
                max_features=1000
            )
        else:  # nmf
            vectorizer = TfidfVectorizer(
                max_df=0.85,
                min_df=2,
                stop_words='english',
                max_features=1000
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
                    'Topic Weight': doc_dist[dominant_topic]
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


@st.cache_resource
def analyze_text(text):
    """Cache the text analyzer initialization"""
    return TextAnalyzer(text)


def main():
    st.title('Text Analysis Tool')

    with st.sidebar:
        st.header("Analysis Options")
        analysis_tab = st.radio(
            "Select Analysis View",
            ["Basic Analysis", "Sentiment Analysis", "Emotion Analysis", "Topic Modeling"]
        )

    # File upload
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])

    if uploaded_file is not None:
        try:
            # Read the file
            text = uploaded_file.getvalue().decode("utf-8")
            with st.expander("View Text Content", expanded=False):
                st.text_area('Text Content', text, height=200)

            # Create analyzer with caching
            analyzer = analyze_text(text)

            # Display basic statistics
            stats = analyzer.basic_statistics()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Words", f"{stats['Total Words']:.0f}")
            with col2:
                st.metric("Total Sentences", f"{stats['Total Sentences']:.0f}")
            with col3:
                st.metric("Avg Word Length", f"{stats['Average Word Length']:.2f}")
            with col4:
                st.metric("Avg Sentence Length", f"{stats['Average Sentence Length']:.2f}")

            if analysis_tab == "Basic Analysis":
                # Word Frequency
                st.header('Word Frequencies')
                col1, col2 = st.columns([1, 2])

                with col1:
                    top_n = st.slider('Select number of top words', 5, 50, 20)
                    freq_df = analyzer.get_word_frequencies(top_n)
                    freq_df.index = freq_df.index + 1
                    st.dataframe(freq_df, use_container_width=True)

                with col2:
                    # Visualization of Word Frequencies
                    try:
                        fig, ax = create_safe_plot()
                        freq_data = freq_df.sort_values('Frequency', ascending=True).tail(15)
                        sns.barplot(x='Frequency', y='Word', data=freq_data, ax=ax)
                        ax.set_title('Word Frequencies')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error creating word frequency plot: {str(e)}")

                # POS Tag Frequencies
                st.header('POS Tag Analysis')
                col1, col2 = st.columns([1, 2])

                with col1:
                    pos_df = analyzer.pos_analysis()
                    st.dataframe(pos_df, use_container_width=True)

                with col2:
                    try:
                        fig, ax = create_safe_plot()
                        pos_data = pos_df.sort_values('Frequency', ascending=True).tail(10)
                        sns.barplot(x='Frequency', y='POS Tag', data=pos_data, ax=ax)
                        ax.set_title('Top 10 POS Tag Frequencies')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error creating POS tag plot: {str(e)}")

                # Tagged Text
                with st.expander("View Tagged Text", expanded=False):
                    try:
                        pos_text = ""
                        for sent in analyzer.doc.sents:
                            for token in sent:
                                pos_text += f"{token.text}[{token.pos_}] "
                            pos_text += "\n\n"
                        st.text_area('Tagged Text', pos_text, height=300)
                    except Exception as e:
                        st.error(f"Error processing tagged text: {str(e)}")

            elif analysis_tab == "Sentiment Analysis":
                st.header('Sentiment Analysis')

                # Overall Sentiment
                sentiment = analyzer.sentiment_analysis()
                polarity = sentiment['Overall Polarity']
                subjectivity = sentiment['Overall Subjectivity']

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sentiment Polarity")
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
                        ax.set_xticklabels(['Negative', 'Somewhat\nNegative', 'Neutral', 'Somewhat\nPositive', 'Positive'])
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error creating polarity chart: {str(e)}")

                    st.metric("Polarity Score", f"{polarity:.3f}",
                              delta=None if -0.1 <= polarity <= 0.1 else ("Positive" if polarity > 0 else "Negative"))

                with col2:
                    st.subheader("Subjectivity")
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
                        ax.set_xticklabels(['Objective', 'Somewhat\nObjective', 'Somewhat\nSubjective', 'Subjective'])
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error creating subjectivity chart: {str(e)}")

                    st.metric("Subjectivity Score", f"{subjectivity:.3f}",
                              delta="Factual" if subjectivity < 0.5 else "Opinionated")

                # Sentence Level Sentiment
                st.subheader('Sentence-Level Sentiment')
                sent_df = analyzer.sentence_sentiment_analysis()

                col1, col2 = st.columns([2, 1])

                with col1:
                    try:
                        fig, ax = create_safe_plot()
                        sentiment_counts = sent_df['Sentiment'].value_counts()
                        colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                        bars = ax.bar(sentiment_counts.index, sentiment_counts.values,
                                     color=[colors.get(x, 'blue') for x in sentiment_counts.index])
                        ax.set_title('Sentence Sentiment Distribution')
                        ax.set_ylabel('Number of Sentences')

                        for i, v in enumerate(sentiment_counts.values):
                            ax.text(i, v + 0.1, str(v), ha='center')

                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error creating sentiment distribution chart: {str(e)}")

                with col2:
                    sentiment_counts_dict = sent_df['Sentiment'].value_counts().to_dict()
                    positive_count = sentiment_counts_dict.get('Positive', 0)
                    negative_count = sentiment_counts_dict.get('Negative', 0)
                    neutral_count = sentiment_counts_dict.get('Neutral', 0)
                    total_sentences = len(sent_df)

                    if total_sentences > 0:
                        st.metric("Positive Sentences", positive_count, f"{positive_count/total_sentences*100:.1f}%")
                        st.metric("Neutral Sentences", neutral_count, f"{neutral_count/total_sentences*100:.1f}%")
                        st.metric("Negative Sentences", negative_count, f"{negative_count/total_sentences*100:.1f}%")

                with st.expander("View Detailed Sentence Analysis", expanded=False):
                    st.dataframe(sent_df, use_container_width=True)

            elif analysis_tab == "Emotion Analysis":
                st.header('Emotion Analysis')

                emotions = analyzer.emotion_analysis()

                st.subheader("Overall Emotion Distribution")

                emotion_df = pd.DataFrame({
                    'Emotion': list(emotions.keys()),
                    'Score': list(emotions.values())
                }).sort_values('Score', ascending=False)

                emotion_colors = {
                    'joy': '#FFCC00',
                    'trust': '#4CAF50',
                    'anticipation': '#FF9800',
                    'surprise': '#8E44AD',
                    'sadness': '#3498DB',
                    'fear': '#607D8B',
                    'anger': '#F44336',
                    'disgust': '#795548'
                }

                colors = [emotion_colors.get(emotion, '#CCCCCC') for emotion in emotion_df['Emotion']]

                try:
                    fig, ax = create_safe_plot()
                    bars = ax.barh(emotion_df['Emotion'], emotion_df['Score'], color=colors)
                    ax.set_xlabel('Score')
                    ax.set_title('Emotion Analysis')

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
                    st.error(f"Error creating emotion chart: {str(e)}")

                if emotions:
                    dominant_emotion = max(emotions, key=emotions.get)
                    dominant_score = emotions[dominant_emotion]

                    st.subheader("Dominant Emotion")
                    st.markdown(f"The dominant emotion in this text is: **{dominant_emotion.title()}** (Score: {dominant_score:.3f})")

                # Emotion radar chart
                st.subheader("Emotion Profile")

                try:
                    categories = list(emotions.keys())
                    values = list(emotions.values())

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
                        st.info("No significant emotions detected in the text.")
                except Exception as e:
                    st.error(f"Error creating radar chart: {str(e)}")

                # Sentence-level emotion analysis
                st.subheader("Sentence-Level Emotion Analysis")
                try:
                    sentence_emotions = analyzer.sentence_emotion_analysis()

                    emotion_columns = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Trust', 'Anticipation']

                    display_limit = min(20, len(sentence_emotions))
                    heatmap_data = sentence_emotions.iloc[:display_limit][emotion_columns]

                    sentence_labels = [f"S{i+1}: {s[:30]}..." if len(s) > 30 else f"S{i+1}: {s}"
                            for i, s in enumerate(sentence_emotions['Sentence'].iloc[:display_limit])]

                    if not heatmap_data.empty and heatmap_data.sum().sum() > 0:
                        fig, ax = plt.subplots(figsize=(12, max(6, display_limit * 0.4)))
                        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu",
                                   yticklabels=sentence_labels, fmt='.2f', ax=ax)
                        ax.set_title('Emotion Distribution Across Sentences')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("No significant emotions detected in individual sentences.")

                    with st.expander("View Detailed Sentence Emotions", expanded=False):
                        cols = ['Sentence', 'Dominant Emotion'] + emotion_columns
                        st.dataframe(sentence_emotions[cols], use_container_width=True)

                    # Dominant emotions distribution
                    st.subheader("Distribution of Dominant Emotions")
                    dominant_counts = sentence_emotions['Dominant Emotion'].value_counts()

                    if len(dominant_counts) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors_for_pie = [emotion_colors.get(emotion.lower(), '#CCCCCC') for emotion in dominant_counts.index]

                        patches, texts, autotexts = ax.pie(
                            dominant_counts,
                            labels=dominant_counts.index,
                            autopct='%1.1f%%',
                            colors=colors_for_pie,
                            startangle=90
                        )

                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(10)

                        ax.axis('equal')
                        ax.set_title('Distribution of Dominant Emotions in Sentences')
                        st.pyplot(fig)
                        plt.close(fig)

                except Exception as e:
                    st.error(f"Error in sentence-level emotion analysis: {str(e)}")

            elif analysis_tab == "Topic Modeling":
                st.header('Topic Modeling Analysis')

                st.markdown("""
                Topic modeling helps identify the main themes or topics in your text.
                This analysis uses **Latent Dirichlet Allocation (LDA)** or **Non-negative Matrix Factorization (NMF)**
                to discover hidden topics based on word patterns.
                """)

                # Check if we have enough text
                if len(analyzer.sentences) < 3:
                    st.warning("Topic modeling requires at least 3 sentences. Your text is too short for meaningful topic analysis.")
                else:
                    # Model selection and parameters
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        method = st.selectbox(
                            "Select Method",
                            ["lda", "nmf"],
                            format_func=lambda x: "LDA (Latent Dirichlet Allocation)" if x == "lda" else "NMF (Non-negative Matrix Factorization)"
                        )

                    with col2:
                        n_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5)

                    with col3:
                        n_words = st.slider("Words per Topic", min_value=5, max_value=20, value=10)

                    # Perform topic modeling
                    try:
                        with st.spinner('Extracting topics...'):
                            topic_results = analyzer.topic_modeling_lda(
                                n_topics=n_topics,
                                n_top_words=n_words,
                                method=method
                            )

                        if topic_results is None:
                            st.error("Could not perform topic modeling. Make sure your text has enough content and variety.")
                        else:
                            topics = topic_results['topics']
                            doc_topic_dist = topic_results['doc_topic_dist']
                            dominant_topics_df = topic_results['dominant_topics']

                            # Display topics
                            st.subheader("Discovered Topics")

                            # Create columns for topics
                            cols_per_row = 2
                            topic_items = list(topics.items())

                            for i in range(0, len(topic_items), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, (topic_name, topic_data) in enumerate(topic_items[i:i+cols_per_row]):
                                    with cols[j]:
                                        st.markdown(f"### {topic_name}")
                                        words_str = ", ".join(topic_data['words'][:8])
                                        st.info(f"**Top words:** {words_str}")

                            # Visualize topics as word clouds/bar charts
                            st.subheader("Topic Word Distributions")

                            for topic_name, topic_data in topics.items():
                                try:
                                    fig, ax = create_safe_plot(figsize=(10, 4))

                                    words = topic_data['words'][:n_words]
                                    weights = topic_data['weights'][:n_words]

                                    # Normalize weights for better visualization
                                    weights = np.array(weights)
                                    weights = weights / weights.sum()

                                    # Create color palette
                                    colors_palette = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))

                                    bars = ax.barh(range(len(words)), weights, color=colors_palette)
                                    ax.set_yticks(range(len(words)))
                                    ax.set_yticklabels(words)
                                    ax.set_xlabel('Weight')
                                    ax.set_title(f'{topic_name} - Top Words')
                                    ax.invert_yaxis()

                                    # Add value labels
                                    for i, (bar, weight) in enumerate(zip(bars, weights)):
                                        width = bar.get_width()
                                        ax.text(width, bar.get_y() + bar.get_height()/2,
                                               f' {weight:.3f}',
                                               va='center', ha='left', fontsize=9)

                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)

                                except Exception as e:
                                    st.error(f"Error creating visualization for {topic_name}: {str(e)}")

                            # Topic distribution across documents
                            st.subheader("Topic Distribution")

                            try:
                                fig, ax = create_safe_plot(figsize=(12, 6))

                                # Create heatmap of document-topic distribution
                                # Limit to first 30 documents for visibility
                                display_limit = min(30, len(doc_topic_dist))
                                heatmap_data = doc_topic_dist[:display_limit]

                                im = ax.imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd')

                                # Set ticks and labels
                                ax.set_yticks(range(n_topics))
                                ax.set_yticklabels([f'Topic {i+1}' for i in range(n_topics)])
                                ax.set_xlabel('Sentence Index')
                                ax.set_ylabel('Topics')
                                ax.set_title('Topic Distribution Across Sentences')

                                # Add colorbar
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('Topic Weight')

                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                            except Exception as e:
                                st.error(f"Error creating topic distribution visualization: {str(e)}")

                            # Show topic distribution pie chart
                            st.subheader("Overall Topic Proportions")

                            try:
                                # Calculate average topic weights across all documents
                                avg_topic_weights = doc_topic_dist.mean(axis=0)

                                fig, ax = plt.subplots(figsize=(8, 8))

                                colors_pie = plt.cm.Set3(range(n_topics))
                                patches, texts, autotexts = ax.pie(
                                    avg_topic_weights,
                                    labels=[f'Topic {i+1}' for i in range(n_topics)],
                                    autopct='%1.1f%%',
                                    colors=colors_pie,
                                    startangle=90
                                )

                                for autotext in autotexts:
                                    autotext.set_color('white')
                                    autotext.set_fontweight('bold')

                                ax.axis('equal')
                                ax.set_title('Average Topic Proportions in Text')
                                st.pyplot(fig)
                                plt.close(fig)

                            except Exception as e:
                                st.error(f"Error creating topic proportions chart: {str(e)}")

                            # Dominant topics per sentence
                            st.subheader("Sentence-Topic Assignments")

                            st.markdown("Each sentence is assigned to its most dominant topic:")

                            # Show statistics
                            topic_counts = dominant_topics_df['Dominant Topic'].value_counts()
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                st.markdown("**Topic Distribution:**")
                                for topic, count in topic_counts.items():
                                    percentage = (count / len(dominant_topics_df)) * 100
                                    st.metric(topic, count, f"{percentage:.1f}%")

                            with col2:
                                try:
                                    fig, ax = create_safe_plot()
                                    colors_bar = plt.cm.Set3(range(len(topic_counts)))
                                    bars = ax.bar(topic_counts.index, topic_counts.values, color=colors_bar)
                                    ax.set_xlabel('Topic')
                                    ax.set_ylabel('Number of Sentences')
                                    ax.set_title('Sentences per Topic')

                                    # Add value labels
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
                                    st.error(f"Error creating topic distribution chart: {str(e)}")

                            # Show detailed sentence-topic mapping
                            with st.expander("View Detailed Sentence-Topic Assignments", expanded=False):
                                # Sort by topic weight
                                sorted_df = dominant_topics_df.sort_values('Topic Weight', ascending=False)
                                st.dataframe(sorted_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error performing topic modeling: {str(e)}")
                        st.info("Try adjusting the number of topics or make sure your text has sufficient content.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.error("Please make sure the file is a valid text file and try again.")

if __name__ == "__main__":
    main()
