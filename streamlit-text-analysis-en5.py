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
# import torch
from textblob import TextBlob
from nrclex import NRCLex
import matplotlib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use a safe backend and font
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a more reliable font
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign issues

# torch.classes.__path__ = []

# Download NLTK resources
#nltk.download('punkt', quiet=True)
#nltk.download('stopwords', quiet=True)
#nltk.download('averaged_perceptron_tagger', quiet=True)

class TextAnalyzer:
    def __init__(self, text):
        """
        Initialize the TextAnalyzer with a given text.
        
        Args:
            text (str): The full text to be analyzed
        """
        # Load English model
        self.nlp = spacy.load('en_core_web_sm')
        # Clean text of problematic characters
        self.original_text = self.clean_text_for_display(text)
        self.doc = self.nlp(self.original_text)
        self.processed_text = self.preprocess_text(self.original_text)
        self.tokens = word_tokenize(self.processed_text)
        self.sentences = sent_tokenize(self.original_text)
        self.textblob = TextBlob(self.original_text)
        self.emotion_analyzer = NRCLex(self.original_text)
    
    def clean_text_for_display(self, text):
        """Clean text of problematic characters that cause font rendering issues"""
        # Remove carriage returns and other problematic characters
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove any other non-printable characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        return text
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))
        return [token for token in self.tokens if token not in stop_words]
    
    def get_word_frequencies(self, top_n=20):
        cleaned_tokens = self.remove_stopwords()
        freq_dist = FreqDist(cleaned_tokens)
        
        freq_df = pd.DataFrame.from_dict(
            dict(freq_dist.most_common(top_n)), 
            orient='index', 
            columns=['Frequency']
        ).reset_index()
        freq_df.columns = ['Word', 'Frequency']
        
        return freq_df
    
    def pos_analysis(self):
        pos_counts = Counter([token.pos_ for token in self.doc])
        
        pos_df = pd.DataFrame.from_dict(
            pos_counts, 
            orient='index', 
            columns=['Frequency']
        ).reset_index()
        
        pos_df.columns = ['POS Tag', 'Frequency']
        pos_df = pos_df.sort_values('Frequency', ascending=False)
        pos_df.index = range(1, len(pos_df) + 1)
        
        return pos_df
    
    def basic_statistics(self):
        return {
            'Total Words': len(self.tokens),
            'Total Sentences': len(self.sentences),
            'Average Word Length': np.mean([len(word) for word in self.tokens]),
            'Average Sentence Length': len(self.tokens) / len(self.sentences) if len(self.sentences) > 0 else 0
        }
    
    def sentiment_analysis(self):
        """
        Analyze the sentiment of the text using TextBlob.
        
        Returns:
            dict: Dictionary containing sentiment polarity and subjectivity
        """
        return {
            'Overall Polarity': self.textblob.sentiment.polarity,
            'Overall Subjectivity': self.textblob.sentiment.subjectivity
        }
    
    def sentence_sentiment_analysis(self):
        """
        Analyze sentiment for each sentence in the text.
        
        Returns:
            pd.DataFrame: DataFrame containing sentences and their sentiment scores
        """
        sentences = []
        polarities = []
        subjectivities = []
        
        for sentence in self.textblob.sentences:
            # Clean sentence for display
            clean_sentence = self.clean_text_for_display(str(sentence))
            sentences.append(clean_sentence)
            polarities.append(sentence.sentiment.polarity)
            subjectivities.append(sentence.sentiment.subjectivity)
        
        sentiment_df = pd.DataFrame({
            'Sentence': sentences,
            'Polarity': polarities,
            'Subjectivity': subjectivities
        })
        
        # Add a categorical sentiment label based on polarity
        def get_sentiment_label(polarity):
            if polarity > 0.1:
                return 'Positive'
            elif polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        
        sentiment_df['Sentiment'] = sentiment_df['Polarity'].apply(get_sentiment_label)
        
        return sentiment_df
    
    def emotion_analysis(self):
        """
        Analyze emotions in the text using NRCLex which provides emotion categories
        based on the NRC Emotion Lexicon.
        
        Returns:
            dict: Dictionary containing emotion scores
        """
        emotions = self.emotion_analyzer.affect_frequencies
        
        # Focus on the core emotions
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
        sentences = sent_tokenize(self.original_text)
        sentence_emotions = []
        
        for sentence in sentences:
            # Clean sentence for display
            clean_sentence = self.clean_text_for_display(sentence)
            emotion_analyzer = NRCLex(clean_sentence)
            emotions = emotion_analyzer.affect_frequencies
            
            # Extract the core emotions
            emotion_dict = {
                'Sentence': clean_sentence,
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
            if any(emotion_scores.values()):  # Check if any emotions were detected
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                emotion_dict['Dominant Emotion'] = dominant_emotion
            else:
                emotion_dict['Dominant Emotion'] = 'Neutral'
                
            sentence_emotions.append(emotion_dict)
        
        sentence_emotion_df = pd.DataFrame(sentence_emotions)
        
        return sentence_emotion_df

def create_safe_plot():
    """Create a matplotlib figure with safe settings"""
    plt.clf()  # Clear any existing plots
    fig, ax = plt.subplots(figsize=(10, 6))
    return fig, ax

def main():
    st.title('Text Analysis Tool')
    
    with st.sidebar:
        st.header("Analysis Options")
        analysis_tab = st.radio(
            "Select Analysis View",
            ["Basic Analysis", "Sentiment Analysis", "Emotion Analysis"]
        )
    
    # File upload
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            text = uploaded_file.getvalue().decode("utf-8")
            with st.expander("View Text Content", expanded=False):
                st.text_area('Text Content', text, height=200)
            
            # Create analyzer
            analyzer = TextAnalyzer(text)
            
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
                        plt.close(fig)  # Close figure to free memory
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
                        doc = analyzer.nlp(analyzer.original_text)
                        pos_text = ""
                        for sent in doc.sents:
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
                    positive_count = len(sent_df[sent_df['Sentiment'] == 'Positive'])
                    negative_count = len(sent_df[sent_df['Sentiment'] == 'Negative'])
                    neutral_count = len(sent_df[sent_df['Sentiment'] == 'Neutral'])
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
                })
                emotion_df = emotion_df.sort_values('Score', ascending=False)
                
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
                    
                    if any(values):  # Only create radar chart if there are non-zero values
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
                    
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.error("Please make sure the file is a valid text file and try again.")
                
if __name__ == "__main__":
    main()