import torch
from tqdm import tqdm
import time
from datetime import timedelta
import json
import os
import numpy as np

def load_embedding(em_path):
    char2vec_file = os.path.join(em_path, 'char2vec_file.mat.npy')
    word2id_file=os.path.join(em_path, 'word2id.npy')

    char2vec_mat=np.load(char2vec_file)
    word2id=np.load(word2id_file).tolist()
    id2word={id:word for id, word in enumerate(word2id)}

    return char2vec_mat,word2id,id2word

def load_embedding_dict(path):
    char2vec_file = '../../../pretrained_models/word2vec/enwiki_model/char2vec_file.mat.npy'
    word2id_file='../../../pretrained_models/word2vec/enwiki_model/word2id.npy'
    #
    # if os.path.exists(char2vec_file):
    #     char2vec_mat=np.load(char2vec_file)
    #     word2id=np.load(word2id_file).tolist()
    #     id2word={id:word for id, word in enumerate(word2id)}
    #
    #
    #     return char2vec_mat,word2id,id2word

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    char2vec_mat = []
    word2id=['<PAD>', '<UNK>']
    id2word={0:'<PAD>',1:'<UNK>'}
    char2vec_mat.append(np.random.normal(size=(200)))
    char2vec_mat.append(np.random.normal(size=(200)))
    for id, line in enumerate(lines):
        split = line.split(" ")
        char=split[0]

        id2word[len(word2id)] = char
        word2id.append(char)
        char2vec_mat.append(np.array(list(map(float, split[1:]))))

    char2vec_mat = np.array(char2vec_mat, dtype=np.float32)

    np.save(char2vec_file, char2vec_mat)
    np.save(word2id_file,word2id)

    return char2vec_mat,word2id,id2word
    

if __name__ == '__main__':
    load_embedding_dict('../../../pretrained_models/word2vec/enwiki_model/word2vec_gensim')
