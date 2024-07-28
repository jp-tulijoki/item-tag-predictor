# Item-tag predictor

This repository contains the source code and the main result files related to my [Master's thesis](https://helda.helsinki.fi/items/215c11d9-0c3e-4007-906c-5e1f26781820). The purpose of the presented model is to enhance item-tag score prediction for movies and books by extracting new language-based features using BERT. The original model can be found [here](https://github.com/Bionic1251/Revisiting-the-Tag-Relevance-Prediction-Problem/tree/main).

The contents of the repository are the following:
* auxiliary: the files used for calculating and processing the results for the thesis
* model: a slightly altered version of the original MLP using the new features
* preprocess data: the files used to process the original data to a form used by the MLP
* results: the item-tag predictions for the best feature combinations for two different approaches for both books and movies and the results for all feature combinations in all approaches
