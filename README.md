# Orchestra
This repository provides images, data and code for organizing images into clusters using the crowd - see our arXiv report [here](http://arxiv.org/abs/1601.02034). 

### Running the Code
```
$ cd src
$ python run_orchestra.py scenes/intelligent_samples --categorization
```
This will output precision, recall, accuracy and number of unclustered items for the scenes dataset. In addition, you'll also see visualization of confusion matrix in data/scenes/intelligent_samples/categorization. 
See `python run_orchestra.py -h` to get information on the command line options