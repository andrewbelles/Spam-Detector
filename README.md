# Spam-Detector

## Data Pipeline 

Credit to Kaggle User jackksoncsie for the Spam Email dataset that I used for my classifier. 

Data follows the following pipeline from csv to training. Data is first split into subject and body, then using the Universal Sentence Encoder (From `tensorflow_hub`), I encoded each row from the csv into 64 bit floating point vectors. These vectors are L2 normalized, then shuffled in order. Using scikit-learn I partition the full dataset into training and test samples, 80% of the dataset is set aside for training samples and the other 20% are set aside for test samples, all while maintaining class balance. 

## Model Architecture 

I use the Universal Sentence Encoder, a pretrained model provided by `tensorflow_hub` to encode data, providing a deep learning backbone to how we extract numeric values from each email. My inference is done by a linear support vector machine. We compute support vectors over a grid of hyperparameters and optimize for the best performing ones on the training samples. This model is stored. Inference then pulls this stored model, makes predictions on the test samples using the model to showcase efficacy. 

## Usage 

Using the emails.csv file (Or any other dataset that is formatted similarly),
+ `text_encoding.py`: Cleans and encodes the dataset to `.npz` training and test files. 
+ `svm.py`: Trains model and saves best hyperparameters on `training_samples`. 
+ `inference.py`: Performs inferences on `test_samples` and provides performance metrics. 

## Future Plans 

I plan on creating a pipeline to read a faux input stream of emails and perform real-time classification of emails. This requires a more robust formatting that works off of the pure html format. Additionally it might require a more robust way of handling larger emails as the token context of USE-4 might not be sufficient. 

## Output, Hyperparameters, and Confusion Matrix for Current Best Model

Outputs from most recent model

|class|precision|recall|f1|support|
|---|---|---|---|---|
|non-spam|0.990|0.982|0.986|872|
|spam|0.943|0.967|0.955|274|
|accuracy|||0.978|1146|
|macro avg|0.966|0.974|0.970|1146|
|weighted avg|0.978|0.978|0.978|1146|

![Confusion Matrix](confusion_matrix.png)
