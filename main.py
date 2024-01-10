from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import spacy
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from collections import Counter

app = FastAPI()

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

sia = SentimentIntensityAnalyzer()

class TextRequest(BaseModel):
  text: str

# def generate_wordcloud(words, title):
#   wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)

#   plt.figure(figsize=(10, 5))
#   plt.imshow(wordcloud, interpolation='bilinear')
#   plt.axis('off')
#   plt.title(title)
#   plt.show()

# def generate_word_count_graph(word_counts):
#   words, counts = zip(*word_counts.items())

#   plt.figure(figsize=(12, 6))
#   plt.bar(words, counts)
#   plt.xlabel('Words')
#   plt.ylabel('Count')
#   plt.title('Word Count Graph')
#   plt.xticks(rotation=45, ha='right')
#   plt.show()

# def generate_heatmap(word_counts):
#   words, counts = zip(*word_counts.items())

#   data = {'Words': words, 'Counts': counts}
#   df = pd.DataFrame(data)

#   plt.figure(figsize=(10, 6))
#   sns.heatmap(df.pivot_table(index='Words', columns='Counts', aggfunc=len).fillna(0), cmap='YlGnBu')
#   plt.title('Word Count Heatmap')
#   plt.show()

@app.get('/')
def __index__():
  return "Hello from Our NER with FastAPI Project"

@app.post("/wordcloud-named-entities", response_class=HTMLResponse)
async def generate_wordcloud(request: TextRequest):
  try:
    # Process the input text with SpaCy
    doc = nlp(request.text)

    # Extract named entities and create a list of words
    named_entities = [ent.text for ent in doc.ents]

    # Join the words into a single string
    text_for_wordcloud = " ".join(named_entities)

    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_for_wordcloud)

    # Save the word cloud to a file (optional)
    # wordcloud.to_file("wordcloud.png")

    # Display the generated word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Return the HTML response with the embedded image
    return HTMLResponse(content=f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" />')

  except Exception as e:
    # Handle any exceptions (e.g., invalid input)
    raise HTTPException(status_code=400, detail=str(e))
  finally:
    # Close the matplotlib plot to avoid memory leaks
    plt.close()

@app.post("/wordcloud-sentiment-analysis", response_class=HTMLResponse)
async def generate_sentiment_wordcloud(request: TextRequest):
  try:
    # Perform sentiment analysis using NLTK
    sentiment_scores = sia.polarity_scores(request.text)
    print(sentiment_scores)
        
    # Extract words based on sentiment (positive or negative)
    words = [word for word in request.text.split() if sentiment_scores['compound'] > 0.1 or sentiment_scores['compound'] < -0.1]
    print(words)

    # Join the words into a single string
    text_for_wordcloud = " ".join(words)

    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_for_wordcloud)

    # Save the word cloud to a file (optional)
    # wordcloud.to_file("wordcloud.png")

    # Display the generated word cloud using matplotlib
    plt.figure(figsize=(20, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Encode the image as base64
    image_base64 = base64.b64encode(buf.read()).decode()

    # Generate a word count graph
    word_count = Counter(words)
    words, counts = zip(*word_count.items())
    plt.figure(figsize=(20, 6))

    # Plot word count graph
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Word Count Graph')
    plt.xticks(rotation=45, ha='right')
    word_count_image_buf = BytesIO()
    plt.savefig(word_count_image_buf, format='png')
    word_count_image_buf.seek(0)
    word_count_image_base64 = base64.b64encode(word_count_image_buf.read()).decode()

    # Generate a heat map
    plt.figure(figsize=(20, 6))
    sns.heatmap([counts], annot=True, fmt='d', cmap='Blues', xticklabels=words, yticklabels=False)
    plt.title('Word Count Heat Map')
    heat_map_image_buf = BytesIO()
    plt.savefig(heat_map_image_buf, format='png')
    heat_map_image_buf.seek(0)
    heat_map_image_base64 = base64.b64encode(heat_map_image_buf.read()).decode()

    # Return the HTML response with the embedded image
    return HTMLResponse(
      content=f'''
        <h2>Word Cloud</h2>
        <img src="data:image/png;base64,{image_base64}" /><br/><br/>
            
        <h2>Word Count Graph</h2>
        <img src="data:image/png;base64,{word_count_image_base64}" /><br/><br/>
            
        <h2>Word Count Heat Map</h2>
        <img src="data:image/png;base64,{heat_map_image_base64}" />
        '''
    )

  except Exception as e:
    # Handle any exceptions (e.g., invalid input)
    raise HTTPException(status_code=400, detail=str(e))
  finally:
    # Close the matplotlib plot to avoid memory leaks
    plt.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)