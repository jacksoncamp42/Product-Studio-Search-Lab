# from transformers import pipeline

# def analyze_sentiment(text):
#     # Explicitly specify the model and device
#     sentiment_pipeline = pipeline(
#         "sentiment-analysis",
#         model="distilbert-base-uncased-finetuned-sst-2-english",
#         device=-1 # 0 for GPU, -1 for CPU - we're not using GPU in this case
#     )
#     result = sentiment_pipeline(text)[0]
#     label = result['label']
#     score = result['score']
#     if label == "NEGATIVE":
#         return 1 - score
#     elif label == "POSITIVE":
#         return score
#     else:
#         raise ValueError("Unexpected sentiment label received")
    
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()
text_1 = "The book was a perfect balance between wrtiting style and plot."
text_2 =  "The pizza tastes terrible."
sent_1 = sentiment.polarity_scores(text_1)
sent_2 = sentiment.polarity_scores(text_2)
print("Sentiment of text 1:", sent_1)
print("Sentiment of text 2:", sent_2)

        
response = """
The **Weill Cornell Medicine’s Center for Reproductive Medicine** has been ranked as the #1 fertility clinic in the nation by Newsweek. The center specializes in helping patients with complex medical histories achieve parenthood and offers comprehensive care with state-of-the-art fertility tests and procedures. Since 1988, more than 31,500 babies have been born as a result of in vitro fertilization (IVF) performed at this center, in addition to tens of thousands of babies born through other treatment modalities. Dr. Zev Rosenwaks, the director and physician-in-chief of the center, emphasizes their commitment to helping aspiring parents realize their dreams even after unsuccessful attempts elsewhere. The center's outstanding performance and dedication to patient care have contributed to its recognition as the top fertility clinic in the nation.

In addition to the top-ranking Weill Cornell Medicine’s Center for Reproductive Medicine, the **NYU Langone Fertility Center** in New York City also offers comprehensive fertility care for families. While not specifically ranked as the #1 fertility clinic in the nation, the NYU Langone Fertility Center provides a range of services for individuals and couples seeking fertility treatments.

Both of these centers have a strong reputation in New York City for their expertise, advanced technologies, and successful outcomes in helping individuals and couples achieve their dreams of starting a family through various fertility treatments.
"""

sentiment_score = sentiment.polarity_scores(response)['pos']
print(sentiment_score)