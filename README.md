# **Calculating the Similarity Between Two Phrases Using Lexical Methods**

## **Project Overview**
This project aims to analyze the similarity between two phrases using various lexical methods, including co-occurrence and TF-IDF. The goal is to calculate similarity scores and evaluate the effectiveness of these methods in quantifying how closely related two phrases are. The project provides insights into how different approaches can affect similarity calculations in various contexts.

---

## **Key Features**

### 1. **Data Loading**:
   - The project begins by loading a dataset containing phrases for similarity analysis.

### 2. **Data Preprocessing**:
   - Text preprocessing steps are applied to ensure that the phrases are clean and uniform for analysis. This includes converting text to lowercase, removing special characters, and applying any necessary transformations.

### 3. **Similarity Calculation**:
   - Various methods are employed to calculate similarity between phrases:
     - **Cosine Similarity**: Measures the cosine of the angle between two vectors, providing a similarity score based on their direction.
     - **Jaccard Similarity**: Assesses the similarity by comparing the size of the intersection to the size of the union of two sets of words.
     - **Euclidean Distance**: Calculates the "straight-line" distance between two points in a multidimensional space, used to assess dissimilarity.

### 4. **Lexical Methods**:
   - The project leverages lexical methods such as co-occurrence and TF-IDF (Term Frequency-Inverse Document Frequency) to evaluate the importance of words in the phrases relative to the entire dataset.

### 5. **Machine Learning Models**:
   - Various machine learning models are implemented to predict the similarity scores based on the features extracted from the phrases using lexical methods. 
   - The project evaluates model performance using metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)** to quantify how well the models perform.

---

## **Project Workflow**

1. **Import Libraries**: Load the necessary libraries for data manipulation, text processing, and machine learning model evaluation.
2. **Load the Dataset**: Import the dataset containing the phrases to analyze.
3. **Preprocess Data**: Apply preprocessing steps to clean and standardize the text.
4. **Calculate Similarities**: Use various methods to calculate the similarity scores between the phrases.
5. **Train Machine Learning Models**: Implement machine learning models to predict similarity scores based on the lexical features extracted.
6. **Evaluate Model Performance**: Assess the models using MSE and R² to understand their accuracy and effectiveness.

---

## **How to Run the Project**

1. **Install Dependencies**: Ensure that all necessary libraries for data manipulation, machine learning, and text processing are installed. This may include libraries such as `pandas`, `numpy`, `scikit-learn`, and `nltk`.
2. **Load Dataset**: Import your dataset containing the phrases for similarity analysis.
3. **Preprocess Data**: Clean and prepare the text for analysis.
4. **Calculate Similarities**: Apply the various similarity calculation methods to analyze the phrases.
5. **Train Models**: Train the machine learning models on the calculated similarity scores and extracted features.
6. **Evaluate Models**: Review the performance metrics to assess the accuracy of the predictions.
