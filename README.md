# Handwritten Letters Recognition
##### Using Machine Learning & PyTorch


This project compares 4 different neural network algorithms on handwritten letters.
1. AlexNet
2. LeNet
3. LeNet (with ReLU)
4. Multi-layer Perceptron

Once the program finishes, it produces a learning curve and also a confusion matrix under the *Evaluation* folder.
These can be used to compare the difference between these algorithms.

Results and Evaluation can be found in the report.



### How to run the project

##### Pre-requisites 
* Download and install Torch module from https://pytorch.org/ into the environment you are working with
* Install sklearn module : "conda install scikit-learn" into anaconda prompt when inside your environment
* Install matplotlib: "conda install -c conda-forge matplotlib" into your environment
* Change the datapath for where the database is downloaded to or want to download it to 
* It can also be downloaded from [here](https://www.nist.gov/itl/products-and-services/emnist-dataset) It also needs to be unzipped if downloaded manually

##### Choosing the model to train and running
* Open the main.py file
* Uncomment the model, you want to train
* And finally, run
