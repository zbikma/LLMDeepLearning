from dotenv import find_dotenv, load_dotenv
import openai
import os
import json
import matplotlib.pyplot  as plt
import tweepy


from langchain.llms import OpenAI

_=load_dotenv(find_dotenv())
openai.api_key=os.getenv('OPENAI_API_KEY')
input_text="I will be waiting here. but you know i hate waiting! so hurry up. I an so sick of this life. i am so depressed."
# Custom prompt with input_text correctly referenced and examples added
custom_prompt = f"""
You are a content moderation assistant. Analyze the following text and classify it based on the following categories:
1. Hate Speech - Example: "I hate all [group]. They are worthless."
2. Violence - Example: "I will hurt you if you come near me."
3. Self-Harm - Example: "I want to hurt myself."
4. Spam - Example: "Click here to win a million dollars!"
5. Misinformation - Example: "The Earth is flat and all scientists are lying."
6. Sarcastic - Example: "Oh great, another Monday. I am thrilled."
7. Depressed - Example: "I feel like nothing matters anymore. Life is pointless."

For each category, indicate whether the text belongs to it ("Yes" or "No"), and assign a score between 0 and 1 for the likelihood.

Please respond in JSON format with the structure:
{{
    "Hate_Speech": {{"flagged": "Yes/No", "score": 0-1}},
    "Violence": {{"flagged": "Yes/No", "score": 0-1}},
    "Self_Harm": {{"flagged": "Yes/No", "score": 0-1}},
    "Spam": {{"flagged": "Yes/No", "score": 0-1}},
    "Misinformation": {{"flagged": "Yes/No", "score": 0-1}},
    "Sarcastic": {{"flagged": "Yes/No", "score": 0-1}},
    "Depressed": {{"flagged": "Yes/No", "score": 0-1}}
}}

Text to analyze: "{input_text}"
"""

# Make the API request
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": custom_prompt}],
    temperature=0,
    top_p=.9
)

# Extract the response content
response_json = response.choices[0].message.content

# Convert the response to a dictionary
try:
    response_dict = json.loads(response_json)
except json.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)
    print("Response content:", response_json)
    exit()

# Debug: Print the response dictionary
print("Response Dictionary:", response_dict)

# Extract categories and scores for visualization, ensuring scores are float
categories = list(response_dict.keys())
scores = [float(response_dict[category]["score"]) for category in categories]

# Debug: Print the categories and scores
print("Categories:", categories)
print("Scores:", scores)

# Plot the bar chart
plt.figure(figsize=(10, 6))
colors = ['grey' if score == 0 else 'skyblue' for score in scores]
bars = plt.bar(categories, scores, color=colors)

# Add annotations for non-zero values
for bar, score in zip(bars, scores):
    if score > 0:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{score:.2f}', 
                 ha='center', va='bottom', fontsize=10, color='black')

plt.xlabel('Moderation Categories')
plt.ylabel('Likelihood Score (0 to 1)')
plt.title('Significance of Moderation Categories')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()