import spacy
from spacy.util import minibatch
from spacy.scorer import Scorer
from tqdm import tqdm
from spacy.training import Example
import random
from Annotated_Dataset import RulerModel, GenerateDataset
import pandas as pd
import os
from Get_Entity_From_File import extract_entities_muc
from Extract_raw_data_tsv import get_raw_text, write_to_file
from matplotlib import pyplot as plt


class NERModel():
    def __init__(self, iterations=2000):
        self.n_iter = iterations
        self.ner_model = spacy.blank("vi")
        self.ner = self.ner_model.add_pipe('ner', last=True)
        self.ner_model.initialize() ###

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
    
def extract_entities_file(folder_path):
    # folder_path = "/path/to/folder"
    allowed_extensions = [".muc"]
    result = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] in allowed_extensions:
            result += extract_entities_muc(file_path)
    return result

# surgery = set(['acute cholangitis', 'appendectomy', 'appendicitis', 'cholecystectomy', 'laser capsulotomy', 'surgisis xenograft', 'sclerotomies', 'tonsillectomy'])
# internalMedicine = set(['asthma', 'atrial fibrillation, congestive heart failure', 'congestive heart failure', 'diabetes emphysema', 'hydrocephalus', 'hyperlipidemia', 'hypertension', 'kidney failure', 'pulmonary embolism', 'stroke', 'urinary tract infection'])
# medication = set(['albuterol inhaler', 'benadryl', 'epinephrine', 'ibuprofen', 'lasix', 'marcaine', 'neurontin', 'pacerone tetracyline', 'tylenol', 'xylicaine', 'zaroxolyn'])
# obstestricsGynecology = set(['abnormal uterine bleeding', 'eclampsia', 'gestational diabetes', 'hysterectomy irregular vaginal bleeding', 'preeclampsia', 'uterine fibroids', 'vaginal hysteretomy', 'vaginal side wall alceration', 'vasomotor symptoms'])
location = []
age = []
date = []
occupation = []
sysmtomanddisease = []
transportation = []
organization = []
person = []

entities = extract_entities_file('/Users/anhduc/Desktop/PPNKKH_DDD/NER_Data/muc_data_folder')

for index in range(len(entities)):
    if(entities[index][0] == 'LOCATION'):
        location.append(entities[index][1])
    elif(entities[index][0] == 'ORGANIZATION'):
        organization.append(entities[index][1])
    elif(entities[index][0] == 'AGE'):
        age.append(entities[index][1])
    elif(entities[index][0] == 'DATE'):
        date.append(entities[index][1])
    elif(entities[index][0] == 'OCCUPATION'):
        occupation.append(entities[index][1])
    elif(entities[index][0] == 'SYSMTOM&DISEASE'):
        sysmtomanddisease.append(entities[index][1])
    elif(entities[index][0] == 'TRANSPORTATION'):
        transportation.append(entities[index][1])
    elif(entities[index][0] == 'PERSON'):
        person.append(entities[index][1])


location = set(location)
age = set(age)
date = set(date)
occupation = set(occupation)
sysmtomanddisease = set(sysmtomanddisease)
transportation = set(transportation)
organization = set(organization)
person = set(person)

raw_df = pd.read_csv("/Users/anhduc/Desktop/PPNKKH_DDD/NER_Data/raw_data.txt", delimiter='\t')
raw_df.head()

rule = RulerModel(location, age, date, occupation, sysmtomanddisease, transportation, organization, person)
annotate = GenerateDataset(rule)
data = annotate.assign_labels_to_documents(raw_df)

model = NERModel()
model.fit(data)
loss_history = [loss['ner'] for loss in model.loss_history]
plt.title("Model training loss history")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(loss_history)
plt.show()

print(model.accuracy_score([('Theo thông báo của Bộ Y tế, từ đầu dịch đến nay, Việt Nam có 10,759 triệu ca mắc Covid-19, đứng thứ 12/227 quốc gia và vùng lãnh thổ.', {'entities': [(19, 21, 'ORGANIZATION'), (22, 23, "ORGANIZATION"), (24, 26, "ORGANIZATION")]})]))


# print(model.loss_history)