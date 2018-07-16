import os
import tensorflow as tf
import json

def make_configs(data_dir, model_dir, model_name):
    configs = {}

    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        raise Exception("QANet data directory (%s) does not exist" % (data_dir))

    model_dir = os.path.abspath(model_dir)
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        raise Exception("QANet model directory (%s) does not exist" % (model_path))

    configs["log_dir"] = os.path.join(model_path, "event")
    configs["save_dir"] = os.path.join(model_path, "model")
    configs["answer_dir"] = os.path.join(model_path, "answer")
    configs["word_emb_file"] = os.path.join(data_dir, "word_emb.json")
    configs["char_emb_file"] = os.path.join(data_dir, "char_emb.json")
    configs["word_dictionary"] = os.path.join(data_dir, "word_dictionary.json")
    configs["char_dictionary"] = os.path.join(data_dir, "char_dictionary.json")

    configs["glove_char_size"] = 94
    configs["glove_word_size"] = int(2.2e6)
    configs["glove_dim"] = 300
    configs["char_dim"] = 64

    configs["para_limit"] = 400
    configs["ques_limit"] = 50
    configs["ans_limit"] = 30
    configs["test_para_limit"] = 1000
    configs["test_ques_limit"] = 100
    configs["char_limit"] = 16
    configs["word_count_limit"] = -1
    configs["char_count_limit"] = -1

    configs["dropout"] = 0.1
    configs["decay"] = 0.9999
    configs["hidden"] = 96
    configs["num_heads"] = 1
    configs["q2c"] = True
    configs["l2_norm"] = 3e-7

    return configs