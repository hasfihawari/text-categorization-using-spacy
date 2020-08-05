import os
import spacy
import pandas as pd
from spacy.util import minibatch

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

def training(train_data):
    nlp = spacy.blank('id')
    
    textcat = nlp.create_pipe('textcat')
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textat']
    with nlp.disable_pipes(*other_pipes):

        nlp.add_pipe(textcat)
        
        textcat.add_label('normal')
        textcat.add_label('fraud')
        textcat.add_label('promo')
        
        optimizer = nlp.begin_training()

        losses = {}
        for epoch in range(10):
            batches = minibatch(train_data, size=5)
            for batch in batches:
                texts, labels = zip(*batch)
                nlp.update(texts, labels, sgd=optimizer, losses=losses)
            print('epoch : {}'.format(epoch))
            
        nlp.to_disk(path_model)
    
if __name__ == '__main__':
    data = load_data()
    training(data)