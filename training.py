import os
import spacy
import pandas as pd
from spacy.util import minibatch
from tqdm import tqdm

path_this = os.path.dirname(os.path.join(__file__))
path_dataset = os.path.abspath(os.path.join(path_this, 'dataset'))
path_model = os.path.abspath(os.path.join(path_this, 'model'))

def load_data():
    data = pd.read_csv(os.path.join(path_dataset, 'dataset.csv'))
    
    data_train = data['Teks'].values
    
    train_label = [
    {
        'cats' : {
        'normal': label == 0, 
        'fraud': label == 1, 
        'promo': label == 2
        }
    } 
    for label in data['label']]
    
    train_data = list(zip(data_train, train_label))
    
    return train_data

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            _gold = gold['cats'][label]
            if label not in gold['cats']:
                continue
            if score >= 0.5 and _gold >= 0.5:
                tp += 1.0
            elif score >= 0.5 and _gold < 0.5:
                fp += 1.0
            elif score < 0.5 and _gold < 0.5:
                tn += 1
            elif score < 0.5 and _gold >= 0.5:
                fn += 1
                
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

def train(train_data, evaluate_data):
    nlp = spacy.blank('id')
    
    if "textcat" not in nlp.pipe_names:
        tc = nlp.create_pipe("textcat")
        nlp.add_pipe(tc)
    else:
        tc = nlp.get_pipe("textcat")

    test_text, test_label = zip(*evaluate_data)
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):
        
        tc.add_label('normal')
        tc.add_label('fraud')
        tc.add_label('promo')
        
        optimizer = nlp.begin_training()

        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        for epoch in range(100):
            losses = {}
            batches = minibatch(train_data, size=32)
            for batch in tqdm(batches):
                texts, labels = zip(*batch)
                nlp.update(texts, labels, drop=0.15, sgd=optimizer, losses=losses)
            with tc.model.use_params(optimizer.averages):
                scores = evaluate(nlp.tokenizer, tc, test_text, test_label)
            print('epoch : {}'.format(epoch+1))
            print("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}"
                  .format(losses["textcat"],scores["textcat_p"],scores["textcat_r"],scores["textcat_f"],))
            
    nlp.to_disk(path_model)
    
if __name__ == '__main__':
    data = load_data()
    data_train = int(len(data) * 0.8)
    data_evaluate = len(data) - data_train
    train(data[:data_train], data[data_evaluate:])
