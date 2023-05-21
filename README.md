## Pain-Classification-Project
This project involves developing a system that can classify pain based on physiological data collected from wearable devices. The system will analyze the provided data and determine whether an individual is experiencing pain or not.

#Experimental Design
The project follows the following steps and requirements:

1. Script: Write a script named Project2.py (or split into multiple files if needed) to implement the system.
2. Data Types: The available data types are Diastolic BP, Systolic BP, EDA, and Respiration.
3. Command-Line Parameters: The script should accept command-line parameters to specify the data type and data file directory. The command format is: python Project2.py <data_type> <data_file>, where data_file is the absolute directory path.
4. Data Collection: The dataset consists of 60 subjects, each providing data for both pain and no pain classes across the available data types. The data is stored in a CSV file with columns: Subject ID, Data Type, Class, and Data. The length of the data can vary.
5. Hand-Crafted Features: Hand-crafted features need to be created. For each data type, calculate the mean, variance, minimum, and maximum values, resulting in 16 features (4 features per data type). Each data instance will have a list of size 4 for individual data types, while a fusion list of size 16 will include features from all data types.
6. Classification: Choose a classifier to train and classify pain vs. no pain based on the extracted features.
7. Cross-Validation: Perform 10-fold cross-validation to train and test the models. Ensure that the same subject does not appear in both the training and testing sets. Each fold consists of 6 subjects, with 9 used for training and 1 for testing. Testing should be conducted for each fold.
8. Performance Evaluation: Print the confusion matrix, classification accuracy, precision, and recall. Average these values across all testing folds. Combine the individual confusion matrices from each fold to obtain the average confusion matrix.
