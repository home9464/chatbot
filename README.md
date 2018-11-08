## A chatbot
This is a slightly modified repo of Pytroch's implementation of chatbot https://pytorch.org/tutorials/beginner/chatbot_tutorial.html



## Download and Install Anaconda3-5.3.0-Linux-x86_64

https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh

## Install "pytorch-1.0"

$conda install pytorch-nightly-cpu -c pytorch

## (optional) Download Cornell Movie Dialog Corpus from here
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

The training data has been saved in the subfolder "data"

## Start training. It will take an hour or less on a moderate PC without GPU
$python train.py


## Start conversation (after training finsihed)
$python chat.py

## Give the bot some personalities?

The training data determine:

1. how many different words does the bot speak (vocabulary)
   
2. how these words are organized to compose a senetence (style)

The 2nd point will be more important because most words are already covered in daily conversations. Since I am a fan of "Friend" (yes, that Friends https://en.wikipedia.org/wiki/Friends).

Would be great if the bot can talk like Joey? Luckily, the full transcript are public available from here: https://fangj.github.io/friends/

So we need to download it, extract all conversations that involved Joey and train the model with it.


