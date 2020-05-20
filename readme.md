

This is the github repository for paper **Latent Alignment of Procedural Concepts in Multimodal Recipes** published in **ALVR2020** ( a ACL2020 workshop). 

# Model overview
![Model overview](https://www.dropbox.com/s/gaa6crusmr4nkbw/model.png?dl=0)
# How to run the program
To start you have to download the images-qa from the [RecipeQA](https://hucvl.github.io/recipeqa/) website and unpack them in the main folder.
Download and move the following image representations to the main folder.
[Test Json ids](https://drive.google.com/file/d/1y3F4baVJ7g_IZkJm4ZyA5eu6ltnyEFDL/view?usp=sharing)
[Train Json ids](https://drive.google.com/file/d/1ARBJ3cLXdEPMC9YXtcmUv39iR2fzUc7j/view?usp=sharing)
[Test Image Representations](https://drive.google.com/file/d/1gu-nxCNQxLI3cyQvHRz6FX6RKNPqZEt2/view?usp=sharing)
[Train Image Representations](https://drive.google.com/file/d/1yiiYaJ2qJgqG-JWaOfCMzYQprLJ5DDLJ/view?usp=sharing)

To run the program you have to run the following code.

    python main.py
You can use the following options.

    -i for number of iterations
    -n for number of samples in use
    -m for the mode of "train" or "test"
    -s for the set  of "train", "test" or "valid"
    -l for using the stored models or not (-l True)
    -c to specify the gpu number
    -p to specify the main folder for the experiment ( Save and load)
    -f to specify the txt file path and name for saving log
    -a for architecture number (7,8,9)
    -e specifying the embedding type ( 1 for bert, 2 for flair, 3 for xlnet)
    -o For specifying the loss mode ("one" for objective 1 and "all" for objective 2)
    -r for specifying the learning rate
    -x for enabling or desabling the modified max pooling

## Dependencies
You have to run Stanford core nlp service at port :9000
Please also install 
` flair, torch, torchvision, PIL, tqdm, pickle, pycorenlp, numpy and math` plugins from python 3.

# How the model works
## image representations
The image representations are the results of the last layer before classification of a resnet50 neural network. 
The network is a pretrained version in torchvision model zoo.
The output of the network for each picture is a 2048 vector representation

## Word Embedding
The word embedding is a pretrained Bert model. We use Flair in order to get the results of the pretraining.

## Pre-proecss of data
### images
in some images the mode is L (grayscale) which gives a different representation from transformers of pytorch. As a result we convert all pictures to RGB before applying resnet on them.
### sentences
we use StanfordCoreNLP to detect sentences from an instruction text body.
### answers
in some cases the answer set contains `' '` which we have to remove.


# Architectures
You have to use `-a 7` for running the experiment with `simple multimodal`, `-a 8` for the experiment with `LXMERT` and `-a 9` for the experiment of unimodal. 

