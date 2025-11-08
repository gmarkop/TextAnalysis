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
        # Load Greek model
        self.nlp = spacy.load('el_core_news_sm')
        self.original_text = text
        self.doc = self.nlp(text)
        self.processed_text = self.preprocess_text(text)
        self.tokens = word_tokenize(self.processed_text)
        self.sentences = sent_tokenize(text)
        self.textblob = TextBlob(text)
        self.emotion_analyzer = NRCLex(text)
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^α-ωΑ-Ωάέήίόύώϊϋΐΰ\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def remove_stopwords(self):
        stop_words = set(stopwords.words('greek'))
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
            'Average Sentence Length': len(self.tokens) / len(self.sentences)
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
            sentences.append(str(sentence))
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
            emotion_analyzer = NRCLex(sentence)
            emotions = emotion_analyzer.affect_frequencies
            
            # Extract the core emotions
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
            if any(emotion_scores.values()):  # Check if any emotions were detected
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                emotion_dict['Dominant Emotion'] = dominant_emotion
            else:
                emotion_dict['Dominant Emotion'] = 'Neutral'
                
            sentence_emotions.append(emotion_dict)
        
        sentence_emotion_df = pd.DataFrame(sentence_emotions)
        
        return sentence_emotion_df

def main():
    st.title('Text Analysis Tool')
    
    with st.sidebar:
        st.header("Analysis Options")
        analysis_tab = st.radio(
            "Select Analysis View",
            ["Basic Analysis", "Sentiment Analysis", "Emotion Analysis"]
        )
    
    # Ανέβασμα αρχείου κειμένου
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file is not None:
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
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Frequency', y='Word', data=freq_df.sort_values('Frequency', ascending=True).tail(15), ax=ax)
                plt.title(f'Top 15 Word Frequencies')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Tag Frequencies
            st.header('POS Tag Analysis')
            col1, col2 = st.columns([1, 2])
            
            with col1:
                pos_df = analyzer.pos_analysis()
                st.dataframe(pos_df, use_container_width=True)
            
            with col2:
                # Εμφάνιση γραφημάτων συχνοτήτων tags
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Frequency', y='POS Tag', data=pos_df.sort_values('Frequency', ascending=True).tail(10), ax=ax)
                plt.title('Top 10 Tag Frequencies')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Tagged Text
            with st.expander("View Tagged Text", expanded=False):
                # Εξαγωγή tags με spaCy
                doc = analyzer.nlp(text)
                
                # Δημιουργία μορφοποιημένου κειμένου με tags
                pos_text = ""
                for sent in doc.sents:
                    for token in sent:
                        pos_text += f"{token.text}[{token.pos_}] "
                    pos_text += "\n\n"
                    
                # Εμφάνιση κειμένου με tags σε ξεχωριστό παράθυρο
                st.text_area('Tagged Text', pos_text, height=300)
        
        elif analysis_tab == "Sentiment Analysis":
            st.header('Sentiment Analysis')
            
            # Overall Sentiment
            sentiment = analyzer.sentiment_analysis()
            polarity = sentiment['Overall Polarity']
            subjectivity = sentiment['Overall Subjectivity']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Polarity")
                # Create a simple gauge chart for sentiment
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 0.1)
                ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                ax.axvspan(-1, -0.1, color='red', alpha=0.2)
                ax.axvspan(-0.1, 0.1, color='gray', alpha=0.2)
                ax.axvspan(0.1, 1, color='green', alpha=0.2)
                ax.scatter(polarity, 0.05, s=300, color='blue', zorder=5)
                
                # Remove y-axis
                ax.set_yticks([])
                ax.set_ylabel('')
                
                # Add labels
                ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                ax.set_xticklabels(['Negative', 'Somewhat\nNegative', 'Neutral', 'Somewhat\nPositive', 'Positive'])
                st.pyplot(fig)
                
                st.metric("Polarity Score", f"{polarity:.3f}", 
                          delta=None if -0.1 <= polarity <= 0.1 else ("Positive" if polarity > 0 else "Negative"))
            
            with col2:
                st.subheader("Subjectivity")
                # Create a simple gauge chart for subjectivity
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 0.1)
                ax.axvspan(0, 0.3, color='lightblue', alpha=0.3)
                ax.axvspan(0.3, 0.7, color='lightgreen', alpha=0.3)
                ax.axvspan(0.7, 1, color='lightyellow', alpha=0.3)
                ax.scatter(subjectivity, 0.05, s=300, color='blue', zorder=5)
                
                # Remove y-axis
                ax.set_yticks([])
                ax.set_ylabel('')
                
                # Add labels
                ax.set_xticks([0, 0.3, 0.7, 1])
                ax.set_xticklabels(['Objective', 'Somewhat\nObjective', 'Somewhat\nSubjective', 'Subjective'])
                st.pyplot(fig)
                
                st.metric("Subjectivity Score", f"{subjectivity:.3f}", 
                          delta="Factual" if subjectivity < 0.5 else "Opinionated")
            
            # Sentence Level Sentiment
            st.subheader('Sentence-Level Sentiment')
            sent_df = analyzer.sentence_sentiment_analysis()
            
            # Show sentiment counts
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Sentence sentiment distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                sentiment_counts = sent_df['Sentiment'].value_counts()
                colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                ax.bar(sentiment_counts.index, sentiment_counts.values, color=[colors.get(x, 'blue') for x in sentiment_counts.index])
                ax.set_title('Sentence Sentiment Distribution')
                ax.set_ylabel('Number of Sentences')
                
                # Add count labels on top of bars
                for i, v in enumerate(sentiment_counts.values):
                    ax.text(i, v + 0.1, str(v), ha='center')
                
                st.pyplot(fig)
            
            with col2:
                # Show metrics
                positive_count = len(sent_df[sent_df['Sentiment'] == 'Positive'])
                negative_count = len(sent_df[sent_df['Sentiment'] == 'Negative'])
                neutral_count = len(sent_df[sent_df['Sentiment'] == 'Neutral'])
                
                st.metric("Positive Sentences", positive_count, f"{positive_count/len(sent_df)*100:.1f}%")
                st.metric("Neutral Sentences", neutral_count, f"{neutral_count/len(sent_df)*100:.1f}%")
                st.metric("Negative Sentences", negative_count, f"{negative_count/len(sent_df)*100:.1f}%")
            
            # Show sentence-level data
            with st.expander("View Detailed Sentence Analysis", expanded=False):
                st.dataframe(sent_df, use_container_width=True)
        
        elif analysis_tab == "Emotion Analysis":
            st.header('Emotion Analysis')
            
            # Get emotion data
            emotions = analyzer.emotion_analysis()
            
            # Overall emotion chart
            st.subheader("Overall Emotion Distribution")
            
            # Create more visually appealing emotion chart
            emotion_df = pd.DataFrame({
                'Emotion': list(emotions.keys()),
                'Score': list(emotions.values())
            })
            emotion_df = emotion_df.sort_values('Score', ascending=False)
            
            # Define a color map for emotions
            emotion_colors = {
                'joy': '#FFCC00',         # Yellow
                'trust': '#4CAF50',       # Green
                'anticipation': '#FF9800', # Orange
                'surprise': '#8E44AD',    # Purple
                'sadness': '#3498DB',     # Blue
                'fear': '#607D8B',        # Blue Grey
                'anger': '#F44336',       # Red
                'disgust': '#795548'      # Brown
            }
            
            # Get colors in the same order as emotions
            colors = [emotion_colors.get(emotion, '#CCCCCC') for emotion in emotion_df['Emotion']]
            
            # Plot horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(emotion_df['Emotion'], emotion_df['Score'], color=colors)
            ax.set_xlabel('Score')
            ax.set_title('Emotion Analysis')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width if width > 0.02 else 0.02
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                        va='center', ha='left' if width <= 0.02 else 'right',
                        color='black' if width <= 0.02 else 'white')
            
            st.pyplot(fig)
            
            # Find dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            dominant_score = emotions[dominant_emotion]
            
            # Display dominant emotion
            st.subheader("Dominant Emotion")
            st.markdown(f"The dominant emotion in this text is: **{dominant_emotion.title()}** (Score: {dominant_score:.3f})")
            
            # Create emotion radar chart with explicit font setting
            st.subheader("Emotion Profile")
    
            # Prepare data for radar chart
            categories = list(emotions.keys())
            values = list(emotions.values())
    
            # Calculate angles for each emotion (excluding the duplication for closing the loop)
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
            # Make the plot circular by appending the first values again
            values = values + [values[0]]
            angles = angles + [angles[0]]
            categories = categories + [categories[0]]  # Also repeat the first category label
        
            # Create radar chart - with explicit font settings
            plt.rcParams['font.family'] = 'Arial'  # Use a simpler font
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
    
            # Plot the radar chart
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
    
            # Set the labels with explicit font properties
            ax.set_xticks(angles[:-1])  # Don't include the duplicate angle
            ax.set_xticklabels(categories[:-1], fontname='Arial', fontsize=10)  # Specify font
    
            # Set y limits
            max_value = max(values)
            ax.set_ylim(0, max_value * 1.1 if max_value > 0 else 0.1)  # Add padding, handle zero case
    
            # Add grid
            ax.grid(True)
    
            st.pyplot(fig)
    
            # Sentence-level emotion analysis
            st.subheader("Sentence-Level Emotion Analysis")
            sentence_emotions = analyzer.sentence_emotion_analysis()
    
            # Create a heatmap of emotions by sentence
            emotion_columns = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Trust', 'Anticipation']
    
            # Limit to first 20 sentences for visualization
            display_limit = min(20, len(sentence_emotions))
            heatmap_data = sentence_emotions.iloc[:display_limit][emotion_columns]
    
            # Create a better sentence representation
            sentence_labels = [f"S{i+1}: {s[:30]}..." if len(s) > 30 else f"S{i+1}: {s}" 
                    for i, s in enumerate(sentence_emotions['Sentence'].iloc[:display_limit])]
    
            # Plot heatmap
            plt.rcParams['font.family'] = 'Arial'  # Use a simpler font
            fig, ax = plt.subplots(figsize=(12, max(6, display_limit * 0.4)))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", yticklabels=sentence_labels, fmt='.2f', ax=ax)
            plt.title('Emotion Distribution Across Sentences')
            plt.tight_layout()
            st.pyplot(fig)
    
            # Show detailed sentence emotions table
            with st.expander("View Detailed Sentence Emotions", expanded=False):
               # Reorder columns to put sentence and dominant emotion first
                cols = ['Sentence', 'Dominant Emotion'] + emotion_columns
                st.dataframe(sentence_emotions[cols], use_container_width=True)
    
            # Show dominant emotions distribution
            st.subheader("Distribution of Dominant Emotions")
            dominant_counts = sentence_emotions['Dominant Emotion'].value_counts()
    
            # Plot pie chart
            plt.rcParams['font.family'] = 'Arial'  # Use a simpler font
            fig, ax = plt.subplots(figsize=(10, 6))
            patches, texts, autotexts = ax.pie(
                dominant_counts, 
                labels=dominant_counts.index, 
                autopct='%1.1f%%',
                colors=[emotion_colors.get(emotion.lower(), '#CCCCCC') for emotion in dominant_counts.index],
                startangle=90
            )
    
            # Make text more readable
            for text in texts:
                text.set_fontsize(12)
            for autotext in autotexts:
                text.set_fontsize(10)
                autotext.set_color('white')
    
            ax.axis('equal')
            plt.title('Distribution of Dominant Emotions in Sentences')
            st.pyplot(fig)
                
if __name__ == "__main__":
    main()