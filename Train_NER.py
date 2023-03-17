import spacy
from spacy.util import minibatch
from spacy.scorer import Scorer
from tqdm import tqdm
import random
from Annotated_Dataset import RulerModel, GenerateDataset
import pandas as pd
import ruler_model as rl

class NERModel():
    def __init__(self, iterations=10):
        self.n_iter = iterations
        self.ner_model = spacy.blank("en")
        self.ner = self.ner_model.add_pipe('ner', last=True)

    def fit(self, train_data):
        for text, annotations in train_data:
            for ent_tuple in annotations.get('entities'):
                self.ner.add_label(ent_tuple[2])
        other_pipes = [pipe for pipe in self.ner_model.pipe_names 
                       if pipe != 'ner']
        
        self.loss_history = []
        
        train_examples = []
        for text, annotations in train_data:
            train_examples.append(Example.from_dict(
               self.ner_model.make_doc(text), annotations))
        
        with self.ner_model.disable_pipes(*other_pipes): 
            optimizer = self.ner_model.begin_training()
            for iteration in range(self.n_iter):
                print(f'---- NER model training iteration {iteration + 1} / {self.n_iter} ... ----')
                random.shuffle(train_examples)
                train_losses = {}
                batches = minibatch(train_examples, 
                  size=spacy.util.compounding(4.0, 32.0, 1.001))
                batches_list = [(idx, batch) for idx, batch in 
                  enumerate(batches)]
                for idx, batch in tqdm(batches_list):
                     self.ner_model.update(
                         batch,
                         drop=0.5,
                         losses=train_losses,
                         sgd=optimizer,
                     )
                 
                self.loss_history.append(train_losses)
                print(train_losses)

    def accuracy_score(self, test_data):
        examples = []
        scorer = Scorer()
        for text, annotations in test_data:
            pred_doc = self.ner_model(text)
            try:
                example = Example.from_dict(pred_doc, annotations)
            except:
                print(f'Error: failed to process document: \n{text},\n\n annotations: {annotations}')
                continue
            
            examples.append(example)
            
        accuracy = scorer.score(examples)
        
        return accuracy
    

surgery = set(['acute cholangitis', 'appendectomy', 'appendicitis', 'cholecystectomy', 'laser capsulotomy', 'surgisis xenograft', 'sclerotomies', 'tonsillectomy'])
internalMedicine = set(['asthma', 'atrial fibrillation, congestive heart failure', 'congestive heart failure', 'diabetes emphysema', 'hydrocephalus', 'hyperlipidemia', 'hypertension', 'kidney failure', 'pulmonary embolism', 'stroke', 'urinary tract infection'])
medication = set(['albuterol inhaler', 'benadryl', 'epinephrine', 'ibuprofen', 'lasix', 'marcaine', 'neurontin', 'pacerone tetracyline', 'tylenol', 'xylicaine', 'zaroxolyn'])
obstestricsGynecology = set(['abnormal uterine bleeding', 'eclampsia', 'gestational diabetes', 'hysterectomy irregular vaginal bleeding', 'preeclampsia', 'uterine fibroids', 'vaginal hysteretomy', 'vaginal side wall alceration', 'vasomotor symptoms'])

raw_df = pd.read_csv('./mtsamples.csv', index_col=0)
raw_df.head()

rule = RulerModel(surgery, internalMedicine, medication, obstestricsGynecology)
annotate = GenerateDataset(rule)
data = annotate.assign_labels_to_documents(raw_df)

model = NERModel()
model.fit(data)