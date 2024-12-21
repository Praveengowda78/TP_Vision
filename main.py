{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "733ea84a-971f-4f44-8d19-b05adf8e47f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From F:\\anaconda\\Lib\\site-packages\\tf_keras\\src\\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Device set to use cpu\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model training and saving completed successfully.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import ast\n",
    "import pickle\n",
    "from datasets import load_dataset\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from sklearn.cluster import KMeans\n",
    "from transformers import pipeline\n",
    "\n",
    "# Load the dataset\n",
    "dataset = load_dataset(\"LDJnr/Puffin\")\n",
    "df = dataset['train'].to_pandas()\n",
    "\n",
    "# Extract human messages from conversations\n",
    "def extract_human_messages(conversations):\n",
    "    if isinstance(conversations, str):\n",
    "        conversations = ast.literal_eval(conversations)\n",
    "    return \" \".join([msg['value'] for msg in conversations if msg['from'] == 'human'])\n",
    "\n",
    "df['human_messages'] = df['conversations'].apply(extract_human_messages)\n",
    "\n",
    "# Generate embeddings using a SentenceTransformer model\n",
    "model = SentenceTransformer('all-MiniLM-L6-v2')\n",
    "embeddings = model.encode(df['human_messages'].tolist())\n",
    "\n",
    "# Perform KMeans clustering with 2 clusters\n",
    "kmeans = KMeans(n_clusters=2, random_state=0)\n",
    "df['topic'] = kmeans.fit_predict(embeddings)\n",
    "\n",
    "# Assign topic labels\n",
    "topic_labels = {0: \"Programming Practices\", 1: \"Misc\"}\n",
    "\n",
    "# Determine which cluster represents \"Programming Practices\"\n",
    "cluster_meanings = {}\n",
    "for cluster_id in range(2):\n",
    "    sample_texts = df[df['topic'] == cluster_id]['human_messages'].head(10)\n",
    "    if any(\"code\" in text.lower() or \"programming\" in text.lower() or \"app\" in text.lower() for text in sample_texts):\n",
    "        cluster_meanings[cluster_id] = \"Programming Practices\"\n",
    "    else:\n",
    "        cluster_meanings[cluster_id] = \"Misc\"\n",
    "\n",
    "# Map the identified labels to the clusters\n",
    "df['topic_label'] = df['topic'].map(cluster_meanings)\n",
    "\n",
    "# Sentiment Analysis: Use a model with three classes (positive, neutral, negative)\n",
    "sentiment_analyzer = pipeline(\"sentiment-analysis\", model=\"distilbert-base-uncased-finetuned-sst-2-english\")\n",
    "\n",
    "# Analyze sentiments\n",
    "def analyze_sentiment(message):\n",
    "    sentiment_result = sentiment_analyzer(message[:512])[0]\n",
    "    return sentiment_result['label']\n",
    "\n",
    "df['sentiment'] = df['human_messages'].apply(analyze_sentiment)\n",
    "\n",
    "# Save the trained models and mappings\n",
    "with open('topic_model.pkl', 'wb') as f:\n",
    "    pickle.dump(kmeans, f)\n",
    "\n",
    "with open('topic_labels.pkl', 'wb') as f:\n",
    "    pickle.dump(cluster_meanings, f)\n",
    "\n",
    "with open('sentence_transformer.pkl', 'wb') as f:\n",
    "    pickle.dump(model, f)\n",
    "\n",
    "print(\"Model training and saving completed successfully.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12b38053-fdf7-4c07-8539-76a4e5a7e140",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfe5da7f-a969-43ab-be59-2d1a6c21002f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
