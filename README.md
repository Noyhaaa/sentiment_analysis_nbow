# Movie Sentiment Analysis with NBOW

This repository contains a simple implementation project of an NBOW (Naive Bag-of-Words) model for sentiment analysis of cinema movies. The model is trained on the IMDb dataset, which consists of movie reviews along with their sentiment (positive or negative). The goal of this project is to provide a basic understanding of setting up a sentiment prediction model.

## How to Use

1. **Prerequisites**: Make sure you have Python installed on your system.

2. **Install Dependencies**: Install the required dependencies by running the following command:
    ```bash
    pip install -r requirements.txt
    ```

3. **Data Preprocessing**: Run the `preprocessing.py` to explore the IMDb dataset and perform data preprocessing.

4. **Model Training**: Use the `build_model.py` to train the NBOW model on the preprocessed data.

5. **Using the Trained Model**: Once the model is trained, you can use it to predict the sentiment of new movie reviews using the functions provided in the `main.py` file.
