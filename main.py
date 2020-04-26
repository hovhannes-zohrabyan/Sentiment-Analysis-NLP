from model.sentiment_analysis_model import SentimentAnalysis

model = SentimentAnalysis()
result, confidence = model.sent_analyze("Hello Dear")
print("Result is " + str(result) + "\n Confidence is " + str(confidence))
