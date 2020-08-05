import os
import spacy
import operator

path_this = os.path.dirname(os.path.join(__file__))
path_model = os.path.abspath(os.path.join(path_this, 'model'))

def load_model():
    nlp = spacy.load(path_model)
    return nlp

def predict(text):
    model = load_model()
    result = model(text)
    result = result.cats
    result = max(result.items(), key=operator.itemgetter(1))[0] 
    
    final_result = {'text': text, 'predict':result}
    return final_result

if __name__ == '__main__':
    text = 'selamat anda mendapatkan hadiah uang tunai 100 juta'
    
    print(predict(text))