import json
import flair
from flair.data import Sentence
import pickle
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, XLNetEmbeddings
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import string
from pycorenlp import StanfordCoreNLP
from os import path
import argparse
import torch.optim as optim
from src.models import LSTMFlair, FullyConnected, Highway, ResidualFullyConnected, HighwayResidualFC, HighwayFC
import random
import numpy as np
import operator
from src.bertModel import NoPosLXRTEncoder
import nltk
import math
from src.single_transformer import TransformerBlock


def parse_arguments(mode="train", number=200, _set="train", load=False, iteration=1, cuda=0, path="saves/", log="saves/log.txt", architecture=1, embedding_type=1, loss_mode="all", learning_rate=0.1, score_mode="max", max_pool=True): 
    parser = argparse.ArgumentParser(description='Getting the arguments passed')
    parser.add_argument('-m','--mode', help='The mode of program',required=False)
    parser.add_argument('-n','--number',help='Number of examples', type=int, required=False)
    parser.add_argument('-i','--iteration',help='Number of iterations', type=int, required=False)
    parser.add_argument('-s','--set',help='Working on which set', required=False)
    parser.add_argument('-l','--load',help='Load or not', type=bool, required=False)
    parser.add_argument('-c', '--cuda', help='Cuda option', type=int, required=False)
    parser.add_argument('-p', '--path', help='Save and Load path', required=False)
    parser.add_argument('-f', '--file', help='Log file name', required=False)
    parser.add_argument('-a', '--architecture', help='Specify Architecture', type=int, required=False)
    parser.add_argument('-e', '--embedding', help='Embedding', type=int, required=False)
    parser.add_argument('-o', '--loss', help='Loss mode', required=False)
    parser.add_argument('-r', '--rate', help='Learning rate', type=float, required=False)
    parser.add_argument('-y', '--score', help='Aggregating Scores Mode', type=str, required=False)
    parser.add_argument('-x', '--maxpool', help='Using customized maxpool', type=bool, required=False, default=True)
    args = parser.parse_args()
    print(args)
    if args.mode and args.mode in ["train", "test"]:
        mode = args.mode
    if args.number:
        number = args.number
    if args.set and args.set in ["test", "train", "valid"]:
        _set = args.set
    if args.load:
        load = args.load
    if args.iteration:
        iteration = args.iteration
    if args.cuda in [0, 1, 2, 3, 4, 5, 6]:
        cuda = args.cuda
    elif args.cuda and args.cuda == -1:
        cuda = -1
    if args.path:
        path = args.path
    if args.file:
        log = args.file
    if args.architecture:
        architecture = args.architecture
    if args.embedding:
        embedding_type = args.embedding
    if args.loss and args.loss in ["random", "all", "one"]:
        loss_mode = args.loss
    if args.rate:
        learning_rate = args.rate
    if args.maxpool:
        max_pool = args.maxpool
    if args.score and args.score in ["max", "mean"]:
        score_mode = args.score
    print(mode, number, _set, load, iteration, cuda, path, log, architecture, embedding_type, loss_mode, learning_rate, score_mode, max_pool)
    
    return mode, number, _set, load, iteration, cuda, path, log, architecture, embedding_type, loss_mode, learning_rate, score_mode, max_pool, args


def read_data(file="train.json"):
    with open(file, 'r') as myfile:
        data = myfile.read()
    # parse file
    info = json.loads(data)
    visual_coherence = [data for data in info['data'] if data['task']=="visual_coherence"]
    textual_cloze = [data for data in info['data'] if data['task']=="textual_cloze"]
    visual_ordering = [data for data in info['data'] if data['task']=="visual_ordering"]
    visual_cloze = [data for data in info['data'] if data['task']=="visual_cloze"]
    print("size of task textual_cloze")
    print(len(textual_cloze))
    print("size of task visual_cloze")
    print(len(visual_cloze))
    print("size of task visual_coherence")
    print(len(visual_coherence))
    print("size of task visual_ordering")
    print(len(visual_ordering))
    print("size of whole set")
    print(len(info['data']))
    return info, visual_cloze, visual_coherence, visual_ordering, textual_cloze

def prepare_data(_set="train"):
    if _set == "train":
        train, train_visual_cloze, train_visual_coherence, train_visual_ordering, train_textual_cloze = read_data(file="train.json")
        return train, train_visual_cloze, train_visual_coherence, train_visual_ordering, train_textual_cloze
    elif _set == "valid":
        valid, val_visual_cloze, val_visual_coherence, val_visual_ordering, val_textual_cloze = read_data(file="val.json")
        return valid, val_visual_cloze, val_visual_coherence, val_visual_ordering, val_textual_cloze
    elif _set == "test":
        test, test_visual_cloze, test_visual_coherence, test_visual_ordering, test_textual_cloze = read_data(file="test.json")
        return test, test_visual_cloze, test_visual_coherence, test_visual_ordering, test_textual_cloze

#Pre-Processing

def sentence_split(text, properties={'annotators': 'ssplit', 'outputFormat': 'json'}):
    """Split sentence using Stanford NLP"""
    annotated = nlp.annotate(text, properties)
    sentence_split = list()
    for sentence in annotated['sentences']:
        s = [t['word'] for t in sentence['tokens']]
        k = [item.lower() for item in s if item not in [",", ".", '...', '..']]
        sentence_split.append(" ".join(k))
    return sentence_split

def preprocess(text):
    text = text.replace("\'\'", "").replace(".", ". ")
    sentences = sentence_split(text)
    results = []
    for sentence in sentences:
        input_str = sentence.lower()
        input_str = input_str.strip()
        input_str = input_str.replace("," , " ,").replace("-rrb-", ")").replace("-lrb-", "(")
        input_str = input_str.split()
        punc = '!"#$%&*+,/:;<=>?@[\]^_`{|}~'
        table = str.maketrans('', '', punc)
        stripped = [w.translate(table) for w in input_str]
        stripped = [w for w in stripped if w]
        results.append(" ".join(stripped))
    return results

#Embedding Fucntion
def embedding(text, embedder):
#    print(text)
    sentence = Sentence(text)
    embedder.embed(sentence)
#    print(sentence)
    return torch.stack([w.embedding for w in sentence])

#getting the vector representation
def get_vector(model, image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 6. Run the model on our transformed image
    data = model(t_img)
    # 8. Return the feature vector
    return data.reshape(2048)

#For language part    
def prepare_language(text, embedder, cuda_option):
    data = embedding(text, embedder)
    data = data.unsqueeze(0)
    if is_cuda:
        data = data.to(device= cuda_option)
    return data

# for pictures part   
def prepare_images(images, cuda_option):
    data = []
    for image in images:
        data.append(get_vector(resnet, image))
    data = torch.stack(data)
    data = data.unsqueeze(0)
    if is_cuda:
        data = data.to(device= cuda_option)
    return data

#For Answer sets    
def prepare_answer(texts, embedder, cuda_option):
    data = []
    for text in texts:
        embed = embedding(text[0], embedder)[-1]
        position = torch.zeros(4)
        position[text[1]] = 1
        if is_cuda:
            position = position.cuda(cuda_option)
        result = torch.cat((embed, position), 0)
#         result = result.squeeze(0)
        data.append(result)
        
    data = torch.stack(data)
    data = data.unsqueeze(0)
    if is_cuda:
        data = data.to(device=cuda_option)
    return data

#execution code for training and testing
def execute(_m, _n, _s, _iteration, _d, base_image_path, log_file, cuda_option, save_path, loss_mode, learning_rate, score_mode, max_pool):
    #set the parameters and optimizer for training
    if _m == "train":
        params = list(answerTransformer.parameters()) + list(contextTransformer.parameters()) + list(imageTransformer.parameters())
        params += list(LSTM_Answer.parameters()) + list(LSTM_Img.parameters()) + list(LSTM_Lang.parameters())
        if architecture == 8:
            params += list(multicoder.parameters()) + list(textTransformer.parameters())
            
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)
    for it in tqdm(range(_iteration)):
        logger = open(log_file, "a+")
        print("-----------------", file=logger)
        logger.write("Start of the iteration "+ str(it) +". \n")
        total_loss = 0
        number_true = 0
        p2 = 0
        passed = 0
        for ind in tqdm(range(0, _n)):
            print("sample number: ", ind, file=logger)
            sample = _d[ind]
            print("sample id: ", sample['recipe_id'], file=logger)
            if _m == "train":
            #zero grad all the learners
                LSTM_Answer.zero_grad()
                LSTM_Img.zero_grad()
                LSTM_Lang.zero_grad()
                contextTransformer.zero_grad()
                answerTransformer.zero_grad()
                imageTransformer.zero_grad()
                if architecture == 8:
                    multicoder.zero_grad()
                    textTransformer.zero_grad()
                
            #Add after having the images
            #prepare the images
            if architecture == 7 or architecture == 8:
                images_list = [image['images'] for image in sample['context']]
                img_data = []
                for info in range(len(images_list)):
                    im_tensor = []
                    check = False
                    for item in images_list[info]:
                        _id = images_id[item]
                        check = True
                        im_tensor.append(torch.from_numpy(images_representation[_id]).to(cuda_option))
                    if check:
#                         print("here")
                        img_data.append(torch.stack(im_tensor))
                    else:
                        img_data.append([])

            #prepare the instructions
            instructions = [text['body'] for text in sample['context']]

            #prepare Question
            placeholder = 0
            g_placeholder = 0
            question = []
            try:
                for q in range(len(sample['question'])):
                    if sample['question'][q] == "@placeholder":
                        placeholder = 1
                        g_placeholder = q
                        continue
                    question.append([sample['question'][q], q])
                question_result = LSTM_Answer(prepare_answer(texts=question, embedder = selected_embedding, cuda_option = cuda_option))[-1][-1]
            except:
                print(q)
                print(sample['question'])
                raise

            #prepare answers pairs
            answers = []
            for item in sample['choice_list']:
                answers.append([item, g_placeholder])
            correct_answer = [sample['choice_list'][sample['answer']], g_placeholder]
            del answers[sample['answer']]
            _list = []
            for _it in range(len(answers)):
                try:
                    if answers[_it][0] == "" or answers[_it][0] == " ":
                        _list.insert(0, _it)
                except:
                    print(_it)
                    raise
            for item in _list:
                del answers[item]
            answers_results = []
            answers_results.append(LSTM_Answer(prepare_answer(texts=[correct_answer], embedder = selected_embedding, cuda_option = cuda_option))[-1][-1])
            for answer in answers:
                 answers_results.append(LSTM_Answer(prepare_answer(texts=[answer], embedder = selected_embedding, cuda_option = cuda_option))[-1][-1])
            answers_results = torch.stack(answers_results)
            answer_results = answerTransformer(answers_results)

            results = []
            try:
                for _it in range(len(instructions)):
                    sentences = preprocess(instructions[_it])
#                     sentences_result = torch.zeros(2048).cuda(cuda_option)
                    if architecture == 7 or architecture == 8 or architecture == 9:
                        all_text = []
                    try:
                        for sentence in sentences:
                            if sentence != "" and sentence != " " and len(sentence) > 3:
                                _input = prepare_language(text= sentence, embedder = selected_embedding, cuda_option = cuda_option)
                                if architecture == 7 or architecture == 8 or architecture == 9:
                                    all_text.append(_input.squeeze(0))
                    except:
                        print(sample)
                        print(sentences)
                        raise
                    if architecture == 7:
                        if not len(all_text):
                            continue
                        all_text = torch.cat(all_text, 0)
                        sentences_result = LSTM_Lang(all_text.unsqueeze(0))[-1][-1]
                        if len(img_data[_it]):
                            image_result = LSTM_Img(img_data[_it].unsqueeze(0))[-1][-1]
                        else:
                            image_result = torch.zeros(2048).cuda(cuda_option)
                    elif architecture == 8:
                        if not len(all_text):
                            continue
                        all_text = torch.cat(all_text, 0)
                        all_text = textTransformer(all_text)
                    #Add after having the images
                        if len(img_data[_it]):
    #                         print("all text shape is: ", all_text.unsqueeze(0).shape)
    #                         print("images shape is: ", img_data[_it].unsqueeze(0).shape)
                            all_text, vision_input = multicoder(lang_feats=all_text.unsqueeze(0),visn_feats=img_data[_it].unsqueeze(0), visn_attention_mask=None, lang_attention_mask=None)
                            sentences_result = LSTM_Lang(all_text)[-1][-1]
                            image_result = LSTM_Img(vision_input)[-1][-1]
                        else:
                            all_text, vision_input = multicoder(lang_feats=all_text.unsqueeze(0),visn_feats=None, visn_attention_mask=None, lang_attention_mask=None)
                            image_result = torch.zeros(2048).cuda(cuda_option)
                            sentences_result = LSTM_Lang(all_text)[-1][-1]

                    if architecture == 9:
                        if not len(all_text):
                            continue
                        all_text = torch.cat(all_text, 0)
                        sentences_result = LSTM_Lang(all_text.unsqueeze(0))[-1][-1]
                        value = contextTransformer(torch.cat((sentences_result, question_result))) 
                    elif architecture == 8 or architecture == 7:
                        value = contextTransformer(torch.cat((sentences_result, question_result, image_result)))
                    else:
                        value = contextTransformer(torch.cat((sentences_result, question_result)))
                    results.append(value)

                results = torch.stack(results)
                a_norm =  answer_results/ answer_results.norm(dim=1)[:, None]
                b_norm = results / results.norm(dim=1)[:, None]
                final_results = torch.mm(a_norm, b_norm.transpose(0,1))
#                 print("final results is: ", final_results, file=logger)
                if max_pool:
                    r11 = final_results.clone()
                    indexes = []
                    for i in range(4):
                        v, i = r11.max(1)
                        v1, i1 = v.flatten().max(0)
                        indexes.append((i1.item(), i[i1.item()].item()))
                        j = torch.arange(r11.size(0)).long()
                        r11[j, i[i1.item()].item()] = -100000000000
                        r11[i1.item()][:] = -100000000000
                    indexes.sort(key = operator.itemgetter(0))
                    index0 = indexes[0][1]
                    index1 = []
                    index2 = []
                    for item in indexes:
                        index1.append(item[0])
                        index2.append(item[1])
                    results = final_results[index1, index2]
                else:
                    results, indexes = final_results.max(1)
                    index0 = indexes[0]
#                 print("the matching maxes are: ", results, file=logger)
                most, index_most = torch.max(results,0)
                print_results = {}
                print_results[sample['answer']] = results[0].item()
                for tt in range(4):
                    if tt == sample['answer']:
                        continue
                    if tt < sample['answer']:
                        print_results[tt] = results[tt+1].item()
                    else:
                        print_results[tt] = results[tt].item()
                print_results_list = []
                for tt in range(4):
                    print_results_list.append(print_results[tt])
                print("the matching result is: ", print_results_list, file=logger)
                print("the predicted answer: ", np.argmax(print_results_list), file=logger)
                print("The answer is: ", sample['answer'], file=logger)
                checking_p2 = torch.tensor(print_results_list).topk(2)[1]
                if sample['answer'] in checking_p2:
                    p2 += 1
                if index_most == 0:
                    number_true += 1
                    print("correct number: ", number_true, file=logger)

                if _m == "train":
                    if loss_mode == "one":
    #                     loss = 1 - results[0]
    #                     ri = random.choice([1, 2, 3])
    #                     for ind in range(final_results[ri].shape[0]):
    #                         loss += max(0, final_results[ri][ind] - 0.2)
                        keys = [1, 2, 3]
                        keys = [key for key in keys if key < final_results.shape[0]]
                        ri = random.choice(keys)
                        loss = 0
                        for key in keys:
                            loss += max(0, final_results[key][index0] - results[0] + 0.1)
                        for ind in range(final_results[ri].shape[0]):
                            loss += max(0, results[ri] - results[0] + 0.1)
                       # print(loss)
                    else:
                        loss = 1- results[0]
                        keys = [1, 2, 3]
                        keys = [key for key in keys if key < final_results.shape[0]]
                        for _it in keys:
                            loss += max(0, results[_it] - 0.1)

#                     print("the loss of this item is: ", loss, file=logger)
                    if loss != 0:
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
            except KeyboardInterrupt:
                logger.close()
                raise
            except:
                raise
                print("GPU PASS")
                passed += 1
                    
        if _m == "train":
            print("total loss is" , total_loss)
            logger.write("The loss of this iteration is "+ str(total_loss) +". \n")
            torch.save(LSTM_Answer.state_dict(), save_path + "ANSWERL")
            torch.save(LSTM_Img.state_dict(), save_path + "IMGL")
            torch.save(LSTM_Lang.state_dict(), save_path + "LANGL")
            torch.save(contextTransformer.state_dict(), save_path + "ContextT")
            torch.save(answerTransformer.state_dict(), save_path + "AnswerT")
            torch.save(imageTransformer.state_dict(), save_path + "ImgT")
            
            if architecture == 8:
                torch.save(multicoder.state_dict(), save_path + "MultiCoder")
                torch.save(textTransformer.state_dict(), save_path + "TextT")
            
            if it % 3 == 0:
                torch.save(LSTM_Answer.state_dict(), save_path + "step" + str(it) + "_ANSWERL")
                torch.save(LSTM_Img.state_dict(), save_path + "step" + str(it) + "_IMGL")
                torch.save(LSTM_Lang.state_dict(), save_path + "step" + str(it) + "_LANGL")
                torch.save(contextTransformer.state_dict(), save_path + "step" + str(it) + "_ContextT")
                torch.save(answerTransformer.state_dict(), save_path + "step" + str(it) + "_AnswerT")
                torch.save(imageTransformer.state_dict(), save_path + "step" + str(it) + "_ImgT")
                
                if architecture == 8:
                    torch.save(multicoder.state_dict(), save_path + "step" + str(it) + "_MultiCoder")
                    torch.save(textTransformer.state_dict(), save_path + "step" + str(it) + "_TextT")
            
        print("The ratio of being correct is: ", number_true / (_n - passed))
        print("The ratio of p2 correct is: ", p2 / (_n - passed), file=logger)
        logger.write("The accuracy is "+ str((number_true / (_n - passed))) +". \n")
        logger.close()
        

def main(mode, number, _set, load, iteration, cuda_option, save_path, log_file, architecture, loss_mode, learning_rate, score_mode, max_pool, args):
    #get the arguments
#     mode, number, _set, load, iteration, cuda_option, save_path, log_file, architecture, args = parse_arguments()
    save_path = str(save_path)
    logger = open(log_file,"a+")
    logger.write("\n --------------- \n Start of the model execution. \n")
    logger.close()
    with open(log_file, 'a+') as f:
        json.dump(args.__dict__, f, indent=2)
    logger = open(log_file,"a+")
    logger.write("\n start the training \n")
    logger.close()
    #prepare the data
    data, data_vcl, data_vc, data_vo, data_tc = prepare_data(_set)
      
    #Transfer to cuda
    if cuda_option in [0, 1, 2, 3, 4, 5, 6] and is_cuda:
        LSTM_Lang.cuda(cuda_option)
        LSTM_Img.cuda(cuda_option)
        LSTM_Answer.cuda(cuda_option)
        contextTransformer.cuda(cuda_option)
        answerTransformer.cuda(cuda_option)
        imageTransformer.cuda(cuda_option)
        flair.device = torch.device(cuda_option) 
        if architecture == 8:
            multicoder.cuda(cuda_option)
            textTransformer.cuda(cuda_option)
    
    #check for the loading parameters
    if load:
        LSTM_Lang.load_state_dict(torch.load(save_path + "LANGL"))
        LSTM_Img.load_state_dict(torch.load(save_path + "IMGL"))
        LSTM_Answer.load_state_dict(torch.load(save_path + "ANSWERL"))
        contextTransformer.load_state_dict(torch.load(save_path + "ContextT"))
        answerTransformer.load_state_dict(torch.load(save_path + "AnswerT"))
        imageTransformer.load_state_dict(torch.load(save_path + "ImgT"))
        if architecture == 8:
            multicoder.load_state_dict(torch.load(save_path + "MultiCoder"))
            textTransformer.load_state_dict(torch.load(save_path + "TextT"))
        
    #set to training
    if mode == "train" and load:
        LSTM_Lang.train()
        LSTM_Img.train()
        LSTM_Answer.train()
        contextTransformer.train()
        answerTransformer.train()
        imageTransformer.train()
        if architecture == 8:
            multicoder.train()
            textTransformer.train()
    #set to the testing
    elif mode == "test" and load:
        LSTM_Lang.eval()
        LSTM_Img.eval()
        LSTM_Answer.eval()
        contextTransformer.eval()
        answerTransformer.eval()
        imageTransformer.eval()
        if architecture == 8:
            multicoder.eval()
            textTransformer.eval()
        
    #define the base address for images
    if _set == "train":
        base_image_path = 'images-qa/train/images-qa/'
    elif _set == "test":
        base_image_path = 'images-qa/test/images-qa/'
    elif _set == "valid":
        base_image_path = 'images-qa/val/images-qa/'
    
    execute(mode, number, _set, iteration, data_tc, base_image_path, log_file, cuda_option, save_path, loss_mode, learning_rate, score_mode, max_pool)

    
# define nlp stanford library
nlp = StanfordCoreNLP('http://localhost:9000')
properties={
  'annotators': 'ssplit',
  'outputFormat': 'json'
  }

#Getting user arguments
mode, number, _set, load, iteration, cuda_option, save_path, log_file, architecture, embedding_type, loss_mode, learning_rate, score_mode, max_pool, args = parse_arguments()


#define the embeddings
if embedding_type == 1:
    selected_embedding = BertEmbeddings()
    embed_dim = 3072
elif embedding_type == 2:
    selected_embedding = FlairEmbeddings("news-forward")
    embed_dim = 2048
elif embedding_type == 3:
    selected_embedding = XLNetEmbeddings()
    embed_dim = 2048
    
bert = BertEmbeddings()
flair = FlairEmbeddings("news-forward")

# Load the pretrained model of resnet
resnet = models.resnet101(pretrained=True)
# Use the model object to select the desired layer
modules = list(resnet.children())[:-1]

resnet = nn.Sequential(*modules)
resnet.eval()

#define the transformers for the picture
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

is_cuda = torch.cuda.is_available()

#define the language part 
LSTM_Lang = LSTMFlair(input_dim=embed_dim, hidden_dim=2048, batch_size = 1)

#define the images LSTM  
LSTM_Img = LSTMFlair(input_dim=2048, hidden_dim=2048, batch_size = 1)

#define the Answers LSTM
LSTM_Answer = LSTMFlair(input_dim=embed_dim+4, hidden_dim=2048, batch_size = 1)  

#define the fully connected instances 
# if architecture == 1:
#     contextTransformer = HighwayResidualFC(size=4096, num_layers=5, f=F.leaky_relu, dims=[4096, 2048, 1024, 512, 512], layers=4)
#     answerTransformer = HighwayResidualFC(size=2048, num_layers=5, f=F.leaky_relu, dims=[2048, 1024, 1024, 512, 512], layers=4)
# elif architecture == 2:
#     contextTransformer = HighwayFC(size=4096, num_layers=5, f=F.leaky_relu, dims=[4096, 2048, 1024, 512, 512], layers=4)
#     answerTransformer = HighwayFC(size=2048, num_layers=5, f=F.leaky_relu, dims=[2048, 1024, 1024, 512, 512], layers=4)
# elif architecture == 3:
#     contextTransformer = ResidualFullyConnected(dims = [4096, 2048, 1024, 512, 512], layers = 4)
#     answerTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)        
# elif architecture == 4:
#     contextTransformer = FullyConnected(dims = [4096, 2048, 1024, 512, 512], layers = 4)
#     answerTransformer = FullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)  
# elif architecture == 5:
#     contextTransformer = ResidualFullyConnected(dims = [2048, 2048, 2048, 512, 512], layers = 4)
#     answerTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
#     imageTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
# elif architecture == 6:
#     contextTransformer = ResidualFullyConnected(dims = [4096, 2048, 2048, 512, 512], layers = 4)
#     answerTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
#     imageTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
    
if architecture == 7:
    contextTransformer = ResidualFullyConnected(dims = [6144, 2048, 2048, 512, 512], layers = 4)
    answerTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
    imageTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
#     lxrt_pooler = BertPooler(hidden_size=embed_dim)

elif architecture == 8:
    multicoder = NoPosLXRTEncoder(visual_feat_dim=2048, drop=0.0, l_layers=3, x_layers=2, r_layers=1, num_attention_heads=4, hidden_size=2048, intermediate_size=2048)
    LSTM_Lang = LSTMFlair(input_dim=2048, hidden_dim=2048, batch_size = 1)
    textTransformer = FullyConnected(dims = [embed_dim, 2048, 2048], layers = 2)
    contextTransformer = ResidualFullyConnected(dims = [6144, 2048, 1024, 512, 512], layers = 4)
    answerTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
    imageTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)

elif architecture == 9:
    contextTransformer = ResidualFullyConnected(dims = [4096, 2048, 2048, 512, 512], layers = 4)
    answerTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
    imageTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
    
if _set == "train":
    images_representation = np.load('train_image_resnet50.npy')
    with open('train_image_resnet50.json', 'r') as f:
        images_id = json.load(f)


elif _set == "test":
    images_representation = np.load('test_image_resnet50.npy')
    with open('test_image_resnet50.json', 'r') as f:
        images_id = json.load(f)
        
main(mode, number, _set, load, iteration, cuda_option, save_path, log_file, architecture, loss_mode, learning_rate, score_mode, max_pool, args)
