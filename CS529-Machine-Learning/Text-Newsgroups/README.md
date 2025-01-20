# CS529-Project2 Text Classification

## Project description

Your task is to implement Naive Bayes and Logistic Regression for document classification as specified in your project description. The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. This collection has become a popular dataset for experiments in text applications of machine learning techniques, such as text classification and text clustering.
Turn in the following:

    1 pdf file containing your project report
    1 zip/rar file containing your commented code and README
    A screenshot of your Kaggle entry (user name/team and accuracy)

Submit your file through UNM Learn. The due date is Midnight (+ 3 hrs buffer) contingent to the late policy stated in the syllabus. Your code should contain appropriate comments to facilitate understanding. If needed, your code must contain a Makefile or an executable script that receives the paths to the training and testing files

## Your report has to be about 6 to 12 pages long and include:

    A high-level description of how your code works.
    The accuracies you obtain under various settings.
    Explain which options work well and why.
    Contrast both algorithms in terms of efficiency and accuracy
    Answers to questions 1 to 8

## Rubric:

    Your code is thoroughly commented (10 pts)
    You provided a well documented README file (10 pts)
    Implementation of Naïve Bayes is correct (15 pts)
    Implementation of Logistic Regression is correct (15 pts)
    Your report is clear, concise and well organized (10 pts)
    Your answers to questions 1 to 7 (40 pts)

TOTAL: 100 pts (10 pts of your final grade)

## Explanation of the Data Set

The data ﬁle (available on UNM Learn and the Kaggle competition) contains four ﬁles:

vocabulary.txt is a list of the words that may appear in documents. The line number is word’s d in other ﬁles. That is, the ﬁrst word (’archive’) has wordId 1, the second (’name’) has wordId 2, etc.

newsgrouplabels.txt is a list of newsgroups from which a document may have come. Again, the line number corresponds to the label’s id, which is used in the .label ﬁles. The ﬁrst line (’alt.atheism’) has id 1, etc.

training.csv Speciﬁes the counts for each of the words used in each of the documents. Each line contains 61190 elements. The first element is the document id, the elements 2 to 61189 are word counts for a vectorized representation of words (refer to the vocabulary for a mapping). The last element is the class of the document (20 different classes). All word/document pairs that do not appear in the ﬁle have count 0.

testing.csv The same as training.csv except that it does not contain the last element.

sample_solution.csv Contains a dummy solution file in the correct format
Data fields

    *position 1 * _- document_id
    *position 2 to position 61189 *_ - count for word xi
    *position 61190 *_- class


## Different Functions and Classes considered for implementation

In this section, we will explain the major functions and classes that are considered for the implementation purpose. 

### Creating the npz dataset

The data/\_\_init__.py file is used for data uploading and conversion purpose. The main class for doing these conversions is Loader. The explanation of the different functions 
in the Loader class is listed below.

* \_\_init__ : Is the constructor of the Loader class
* set_root : Is used to determine the root directory
* \_create_directory : Is used to create the directory
* load : Is used to load the files. The loading of the file is done in both the format: csv and npz
* save : Is used to save the outcome in the csv or npz format
* \_save_csv : Is used to save the file in the csv format
* \_load_csv : Is used to load the csv file
* \_save_npz : Is used to save the data in npz format which is used to reduce the sparsity
* \_load_npz : Is used to load the data in the npz format which is used to reduce the sparsity
* get_sparse : This is used to convert the ndarry into csr to reduce the sparcity

### Creating the Naive Bayes Classifer

The scripts stat/\_\_init__.py, model/\_\_init__.py, and NaiveBayesClassifier.py are used for creating the Naive Bayes Classifier and the description 
of the functions are listed below.

#### stat/\_\_init__

The class MaximumLikelyhoodEstimator is used to create the MLE

* \_\_init__ : Is the constructor of the MaximumLikelyhoodEstimator class
* set_matrix: Is used to generate the matrix from the outcome of the function mle()
* mle : This function is used to calculate the MLE based on the equation that was provided

The class MaximumAPosterioriEstimator is used to construct the MAP

* \_\_init__ : Is the constructor of the MaximumAPosterioriEstimator class
* set_matrix : Is used to generate the matrix from the outcome of the function map()
* map : This function is used to calculate the MAP based on the equation that was provided

#### model/\_\_init__

The class NaiveBayesClassifier is used for the prediction purpose 

* set_mlle : Is used to get the outcome of the mle()
* set_mlap : Is used to get the outcome of the map()
* classify : This function is used for the prediction purpose

#### NaiveBayesClassifier

This script is used for the input parameters on the terminal. The explanation of running this script is done in the later section. The script takes in the outcome of the 
mle() and map() then train the classifier. Then later on that classifer is used for the prediction purpose using the test npz file. 

### Creating the Logisitc Regression

The scripts stat/\_\_init__.py, model/\_\_init__.py, and LinearRegressorClassifier.py are used for creating the Naive Bayes Classifier and the description 
of the functions are listed below.

#### stat/\_\_init__

The class Delta is used to construct the delta matrix

* create\_delta\_matrix: This function used to create the delta matrix from the given input dataset and labels 

#### model/\_\_init__

The class LinearRegressorClassifier is used for creating the Logisitc Regression classifier.

* set\_min\_class : Used for determining the minimum value of the class labels
* set\_eval\_data : The function that takes a portion of the training dataset for validation purpose
* set\_train\_data : The function will take a portion of the training dataset for training the classifier
* logits : Feed forward operation. Computes the prediction values based on current weights. It is activated by a soft_max activation function.
* train : Single training epoch. Updates the weights based on current or given training samples.
* eval : Computes the Mean Square Error of prection using current weights.
* classify : Classifies given samples.

#### LinearRegressorClassifier

This script is used for the input parameters on the terminal. The explanation of running this script is done in the later section. 
Later on that classifer is used for the prediction purpose using the test npz file. 

## Instructions to run the code

All the required are dealt individually and the explanation of running each of the portions are shown below with examples

### Generating the npz files from the given input dataset

The main.py script is used to generate the npz or csv file format for the input file.
<br />
<b>Command:</b> python3 main.py [save-npz] [data\_file] [output\_file]
<br />
<b>Example:</b> python3 main.py save-npz testing.csv data/
<br /><br />


### Training the Naive Bayes Classifier

The following command is used to create the Naive Bayes Classifer
<br />
<b>Command:</b> python3 NaiveBayesClassifier.py train input\_file beta network\_name [output_dir]
<br />
<b>Example:</b> python3 NaiveBayesClassifier.py train data/training.npz 0.0001 test_network TrainedNetworks/
<br /><br />
If train function has no output\_dir, then '.' directory is used instead. 
<br />
The output files are <output_dir>/<network_name>\_beta\_mle.npz and <output_dir>/<network\_name>\_beta\_map.npz
<br />
The predict function will look in the saved\_network\_directory and looks for files <network\_name>\_mle.npz and <network\_name>\_map.npz

### Testing the Naive Bayes Classifer

The following command is used to test the Naive Bayes Classifier
<br />
<b>Command:</b> python3 NaiveBatesClassifier.py predict input\_file network\_name saved\_network\_directory
<br />
<b>Example:</b> python3 NaiveBayesClassifier.py predict data/testing.npz test\_network\_0.0001 TrainedNetworks/

The output will generate the csv file where the header "id" and "class" are included before including into Kaggle. 

### Training the Logisitic Regression Classifier

The following command is used to create the Logisitc Regression Classifier
<br />
<b>Command:</b> python3 LinearRegressorClassifier.py train input\_file eta lamdba epoch max\_tries split network\_name [output_dir]
<br />
<b>Example:</b> python3 LinearRegressorClassifier.py train data/training.npz 0.001 0.001 20000 0.20 test_network TrainedNetworks/

* eta : the learning rate
* lambda : the penalty rate
* epoch : the number of iterations
* max\_tries : the number added up above epoch if the best weight not found
* split : the split betweent the training and validation set

The output files are:
* <output_dir>/<network_name>\_eta\_lambda\_epoch\_weights.npz - The file will provide the weights till the last iteration
* <output_dir>/<network_name>\_eta\_lambda\_epoch\_best\_weights.npz - The file will provide the best weights
* <output_dir>/<network_name>\_eta\_lambda\_finaleval.csv - Contains the outcome of the evaluation
* <output_dir>/<network_name>\_eta\_lambda.log - This file records the loss values in each epoch

### Testing the Logisitic Regression Classifier

The following command is used to test the Logisitc Regression Classifier
<br />
<b>Command:</b> python3 LinearRegressorClassifier.py predict input\_file network\_name saved\_network\_directory
<br />
<b>Example:</b> python3 LinearRegressorClassifier.py predict data/testing.npz test_network\_0.001\_0.001\_20000\_best TrainedNetworks/ 

The output will generate the csv file where the header "id" and "class" are included before including into Kaggle. 

### Extracting of the top 100 words

The following command is used to create the Logisitc Regression Classifier
<br />
<b>Command:</b> python3 NaiveBayesClassifier.py rank network\_name network\_dir [vocabulary_file]
<br />
<b>Example:</b> python3 NaiveBayesClassifier.py rank test\_network\_Q7\_1.6343073805e-05 TrainedNetworks vocabulary.txt
