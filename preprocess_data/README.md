# Preprocess data

This folder contains the files that are used for data pre-processing in the order they should be used. The raw data can be downloaded for movies [here](https://grouplens.org/datasets/movielens/tag-genome-2021/)
and for books [here](https://grouplens.org/datasets/book-genome/).

* get_all_reviews.py: splits the reviews.json in the raw data to ten csv files that can be used in the bert_masked_lm
* bert_masked_lm.py: train a masked language model with the files split previously, one split at a time
* get_review_sample.py: get a sample from all reviews that is used for extracting embeddings
* get_titles.py: get titles to csv that can be used for extracting embeddings
* get_embeddings_for_reviews.py, get_embeddings_for_titles.py, get_tag_embeddings.py: get embeddings for review sample, titles and tags using the previously trained mlm
* get_reviews_embeddings_with_sentence_transformer.py, get_tag_embeddings_with_sentence_transformer.py, get_tag_embeddings_with_sentence_transformer.py: same as above, but with
  a pre-trained model
* cosine_similarity.py: count part of the cosine similarity_features using the embeddings extracted previously. Change mean aggregation manually to max when counting cosine
  similarity between tags and reviews to get both features
* cosine_tag_review.py: count rest of the cosine similarity features. These are attached directly to the processed tenfolds in the datasets
* combine_files.py: combine features counted in cosine_similarity.py to the datasets formed in cosine_tag_review.py to form final datasets for the model   
