IndiaAI - AI for Citizen Safety

This project focuses on classifying crime complaints into predefined categories using a fine-tuned BERT model. The implementation includes text preprocessing, dataset preparation, model training, and saving necessary components for deployment.


Table of Contents: 
Project Overview
Features
Requirements
Usage
Implementation Details
Files and Directories
Model Training Steps
Contact


Project Overview
Crime complaints often contain unstructured data. This project utilizes BERT, a pre-trained language model, to classify textual complaints into appropriate categories effectively.

The following key steps are undertaken:
Data Preprocessing: Clean and normalize the complaint text.
Model Fine-Tuning: Fine-tune the bert-base-uncased model for sequence classification.
Deployment Ready Outputs: Export the trained model, tokenizer, and label encoder.
Features

Text Cleaning: Handles raw complaint text for training.

Robust Classification: Utilizes BERT for natural language understanding.

Imbalanced Dataset Handling: Includes careful training practices to ensure balanced performance.

Deployment Support: Exports trained components for seamless integration into applications.
Requirements

Ensure the following libraries and tools are installed:
pandas
scikit-learn
torch
transformers
joblib
Install Dependencies


Run the following command to install all dependencies:
pip install -r requirements.txt

Usage
1. Prepare Dataset
Ensure your dataset (merged.csv) is formatted with the following columns:

crimeaditionalinfo: Complaint text.
category: Corresponding category label.
Place the merged.csv file in the root directory of the project.

2. Run the Training Script
Execute the training script to preprocess data, train the model, and save outputs:
python train_model.py

4. Outputs
Trained Model: crime_complaint_model_bert.pkl
Tokenizer: tokenizer_bert.pkl
Label Encoder: label_encoder.pkl
Logs and Checkpoints: Saved in the ./results directory.

Implementation Details
1. Preprocessing
Text Cleaning: Converts text to lowercase, removes numbers, punctuation, and excessive whitespace.
Label Encoding: Transforms categorical labels into numeric format using LabelEncoder.

3. Model Training
Model: BertForSequenceClassification from the transformers library.
Tokenization: Input text tokenized using BertTokenizer with padding and truncation to a maximum length of 128 tokens.
Trainer API: Hugging Face's Trainer API is used for training and evaluation.

5. Evaluation
Performed at the end of each epoch, with results logged for further analysis.

Files and Directories

Files
train_model.py: Script to preprocess data, train the model, and save outputs.
merged.csv: Dataset used for training.

Directories
./results: Model checkpoints and evaluation logs.
./trained_model: Final model and tokenizer saved for deployment.
Model Training Steps

1. Data Loading
Load the merged.csv file.
Handle missing values by removing rows where crimeaditionalinfo or category columns are null.

3. Data Preprocessing
Complaint Text: Clean and normalize the text data.
Category Encoding: Convert categories into numeric labels using LabelEncoder.

4. Dataset Preparation
Split the dataset into training and validation sets.
Implement a custom Dataset class to tokenize input and return tensors for training.

5. Model Configuration
Fine-tune a BertForSequenceClassification model using the Trainer API.

6. Training
Train the model for 3 epochs with a batch size of 16.
Save checkpoints and logs for monitoring.

7. Saving Outputs
Export the trained model, tokenizer, and label encoder for deployment.


Contact
For further information or assistance, please reach out at:
Email: cloudopenstack@gmail.com
