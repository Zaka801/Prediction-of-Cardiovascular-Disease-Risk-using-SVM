#SVM-based Cardiovascular Disease Risk Prediction
This repository contains code that utilizes Support Vector Machines (SVM) to predict the risk of cardiovascular disease using the Framingham Heart Study dataset. The code demonstrates data preprocessing, model training, prediction, and evaluation using the scikit-learn library in Python.

#Dataset
The code expects the dataset file framingham.csv to be present in the same directory. The dataset contains various demographic, behavioral, and medical risk factors collected from individuals participating in the Framingham Heart Study.

#Dependencies
The following Python libraries are required to run the code:

pandas
scikit-learn (sklearn)
matplotlib
Install the dependencies using pip:


pip install pandas scikit-learn matplotlib
#Usage
Clone the repository or download the code files.
Place the framingham.csv dataset file in the same directory as the code files.
Run the code using a Python interpreter (e.g., Jupyter Notebook, Anaconda, or any Python IDE).
The code will perform the following steps:

#Load and visualize the dataset using histograms.
Explore and preprocess the data, including handling missing values.
Split the data into training and testing sets.
Create an SVM classifier and train the model using the training data.
Make predictions on the testing data and evaluate the model's accuracy.
Display a comparison of actual and predicted values.
#Results
The code provides an accuracy score that indicates the model's performance in predicting the risk of cardiovascular disease. Additionally, it generates a DataFrame (score) that compares the actual and predicted values for a subset of the testing data.

#Contributing
Contributions to the code, bug fixes, and improvements are welcome. Feel free to submit a pull request or open an issue for any suggestions or problems encountered.

Acknowledgments
The code is based on the scikit-learn library examples and the Framingham Heart Study dataset.
Credits to the original authors of the Framingham Heart Study dataset.
