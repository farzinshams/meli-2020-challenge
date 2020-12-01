import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
from collections import Counter
import dateutil.parser
from datetime import datetime, timedelta
# import json_lines

# i = []
# with open('item_data.jl', 'rb') as f:
#     for item in tqdm(json_lines.reader(f)):
#         i.append(item)


# lr = 0.05
# lr_decay = 0.98
# # params = {
# #     'objective': 'binary',
# #     'metric': 'auc', 
# #     'boosting_type': 'gbdt',
# #     'is_unbalance': True,
# #     'learning_rate': lr,
# #     'num_leaves': 600,
# #     'max_bin': 255, #255,
# #     'feature_fraction': 1,
# #     'subsample': 0.6,
# #     'lambda_l2': 0, #0.65,
# #     'lambda_l1': 0, #0.65,
# #     'num_threads': 58,
# #     'num_boost_round': 80,
# # #     'early_stopping_rounds': 5
# # }

# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'num_leaves': 16,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'num_boost_round': 200,
#     'verbose': 1
# }

# dtrain = lgb.Dataset(dpu_x_train, dpu_y_train)
# dvalid = lgb.Dataset(dpu_x_test, dpu_y_test, reference=dtrain)

# bst = lgb.train(
#     params, dtrain, valid_sets=dvalid, verbose_eval=10,
#     callbacks=[lgb.reset_parameter(learning_rate=lambda current_round: lr*(lr_decay**current_round))],
# )


def to_dict(a, b):
    return {a[i]: b[i] for i in range(len(a))}



def counter(v):
    c = Counter()
    for i in v:
        c[i] += 1
    return c



def get_item_score(item, domain):
    try:
        index = domain_inverse_sorted[domain].index(item)
        return domain_inverse_sorted_score[domain][index]
    except:
        return 0
    
def get_domain_score(item_domain, preds, proba):
    try:
        index = preds.index(item_domain)
        return proba[index]
    except:
        return 0
    
def unique(seq):
    seen = set()
    seen_add = seen.add
    return np.array([x for x in seq if not (x in seen or seen_add(x))])

    

def get_recurrent_datapoint(v):
    c = {}
    for idx, i in enumerate(v):
        if i in c:
            c[i].append(idx)
        else:
            c[i] = [idx]
    
    scores = {}
    for k, l in c.items():
        first = l[0]
        last = l[-1]
        missing = 0
        for i in range(1, len(l)):
            if l[i] != l[i-1] + 1:
                missing += 1
            
        score = missing * len(l) #número de faltantes x comprimento
        scores[k] = score
  
    return scores
#     print(list(c.keys()))
#     target = list(c.keys())[np.argmax(scores)]
#     score = max(scores)
    
#     return target, score


def get_time_score(uh):
    c = {}
    for idx, i in enumerate(uh):
        if i['event_type'] == 'view':
            event_info = i['event_info']
            timestamp = i['event_timestamp']
            if event_info in c:
                c[event_info].append(timestamp)
            else:
                c[event_info] = [timestamp]
    
    scores = {}
    for k, v in c.items():
        scores[k] = (dateutil.parser.parse(v[-1]) - dateutil.parser.parse(v[0])).total_seconds()
        
    return scores

def get_average_condition(v):
    avg_condition = 0
    for i in v:
        avg_condition += item_info[i][1]
    avg_condition /= len(views)
    return avg_condition
    

def undersample(datapoints, ratio=3):
    positive_index = list(datapoints.query('target == 1').index)
    negative_index = list(datapoints.query('target == 0').index)
    negative_fraction = len(positive_index)*ratio
    negative_index_sample = list(np.random.choice(negative_index, negative_fraction))
    return datapoints.loc[positive_index + negative_index_sample].sample(frac=1.)


# def dp_fe(df):
#     df['perc'] = df['count']/df['lenn']
#     df['time_perc'] = df['time_score']/df['total_time']
#     df['prob_item_domain'] = df['prob_domain']*df['prob_item']
#     df['prob_item_text'] = df['prob_text']*df['prob_item']
#     return df

def get_csr_matrix(list_of_lists):
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in list_of_lists:
        for term in d:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), dtype=int)



def average_NDCG(preds, targets, domain):
    
    def relevance(k, l):
        if k == -1:
            return 0
        if k == l:
            return 12
        elif domain[k] == domain[l]:
            return 1
        else:
            return 0
        
    #iDCG = sum(np.array([12,1,1,1,1,1,1,1,1,1])/np.array([np.log2(i + 1) for i in range(1, 11)]))
    iDCG = 15.543559338088349
    
    NDCGs = []
    for i in tqdm(range(len(preds))):
        pred = preds[i]
        target = targets[i]
        
        assert len(pred) == 10, print(i)
        assert len(set(pred)) == 10, print(i)
        
        DCG = 0
        for idx, k in enumerate(pred):
            DCG += relevance(k, target)/np.log2(1 + (idx + 1))

        NDCGs.append(DCG/iDCG)
    
    return np.mean(NDCGs)
