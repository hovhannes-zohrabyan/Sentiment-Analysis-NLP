from controller.sentiment_analysis_model import SentimentAnalysisModelCreation

if __name__ == '__main__':
    sentiment_analysis_model = SentimentAnalysisModelCreation()
    sentiment_analysis_model.train()
