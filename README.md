# Pytorch_Naruto_Hand_Signs
Used Pytorch with transfer learning to make a naruto hand sign recognizer. I have used resnet18 model to train this classifier and only trained it for 5 epochs.

This repository contains also a python script which launch your webcam so you can try it yourself.

# Get started
You can download the dataset and launch the entier colaboratory which will train the model and save it to the ./models folder then you can launch this command to try it yourself : 

```
python camclass.py ./path/to/model.pth
```

# More information
The main weakness of this dataset is that all the hand sign are centered, so whanever you're moving your hands far from the center, the model doesn't predict correctly.

# Dataset
Link to the dataset: https://www.kaggle.com/vikranthkanumuru/naruto-hand-sign-dataset

# TODOs:
* Create my own dataset with variant position
* Experience other models
