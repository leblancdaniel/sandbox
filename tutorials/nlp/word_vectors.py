import os
import numpy as np
import pandas as pd
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Load the large model to get the vectors
nlp = spacy.load("en_core_web_lg")
# Load data
script_dir = os.path.dirname(__file__)
rel_path = "yelp_ratings.csv"
filepath = os.path.join(script_dir, rel_path)
review_data = pd.read_csv(filepath)

# We just want the vectors so we can turn off other models in the pipeline
with nlp.disable_pipes():
    vectors = np.array([nlp(review.text).vector for idx, review in review_data.iterrows()])

X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment, 
                                                    test_size=0.1, random_state=1)
# Create the LinearSVC model
model = LinearSVC(random_state=1, dual=False)
model.fit(X_train, y_train)
# Uncomment and run to see model accuracy
#print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')

review = """I absolutely love this place. The 360 degree glass windows with the 
Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
transports you to what feels like a different zen zone within the city. I know 
the price is slightly more compared to the normal American size, however the food 
is very wholesome, the tea selection is incredible and I know service can be hit 
or miss often but it was on point during our most recent visit. Definitely recommend!

I would especially recommend the butternut squash gyoza."""

review_vec = nlp(review).vector
## Center the document vectors
# Calculate the mean for the document vectors, should have shape (300,)
vec_mean = vectors.mean(axis=0)
# Subtract the mean from the vectors
centered = vectors - vec_mean
# Calculate similarities for each document in the dataset
def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))
# Make sure to subtract the mean from the review vector
sims = np.array([cosine_similarity(review_vec - vec_mean, vec) for vec in centered])
# Get the index for the most similar document
most_similar = sims.argmax()
# Print most similar entry in review_data to review text above
print(review_data.iloc[most_similar].text)
