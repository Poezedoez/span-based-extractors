import json
from model.util import split
import os

def reformat_conll03(input_path, output_path):
    dataset = []
    tokens = []
    entities = []
    relations = []
    entity = {'type':'O', 'start':0, 'end':0}
    previous_label = 'O'

    def add_entity(entity):
        entity['end'] = len(tokens)
        if entity['type'] != 'O':
            entities.append(entity)

    def create_entity(label):
        type_ = label.split('-')[1] if label != 'O' else 'O'
        entity = {'type':type_, 'start':len(tokens)}

        return entity
    
    with open(input_path, 'r', encoding='utf-8') as in_file:
        for line_number, line in enumerate(in_file):
            ## No data yet
            if line == '\n' and not tokens:
                continue

            ## Empty line means end of sentence
            if line == '\n':
                add_entity(entity)
                dataset.append({'tokens':tokens, 'entities':entities, 'relations':relations, 'orig_id':line_number})
                tokens = []
                entities = []
                relations = []
                entity = create_entity('O')
                continue

            token, _, _, label = line.strip().split(" ")
            
            ## End of previous entity span
            if label != previous_label:
                add_entity(entity)
                entity = create_entity(label)

            tokens.append(token)
            previous_label = label

    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(dataset, out_file)

def reformat_semeval2017task10(input_path, output_path):
    
    def parse_entity(line, char2word_position):
        type_, start, end = line[1].split(' ')
        start = char2word_position[int(start)]
        end = char2word_position[int(end)-1] + 1
        entity = {'type':type_, 'start':start, 'end':end}

        return entity

    def parse_relation(line):
        try:
            relation, head, tail = line[1].split(' ')
        except:
            relation, head, tail = line[1].split(' ')[0], line[1].split(' ')[1], line[1].split(' ')[2]
        head = head if line[0] == '*' else head.split(':')[1]
        tail = tail if line[0] == '*' else tail.split(':')[1]
        relation = {'type':relation, 'head':indicator2index[head], 'tail':indicator2index[tail]}

        return relation
    
    dataset = []
    flist = os.listdir(input_path)
    for f in flist:
        if not f.endswith(".ann"):
            continue
        f_anno = open(os.path.join(input_path, f), "r", encoding='utf-8')
        f_text = open(os.path.join(input_path, f.replace(".ann", ".txt")), "r", encoding='utf-8')

        # Text paragraph is on one line
        text = f_text.readline()
        tokens, char2word_position = split(text)
        entities = []
        indicator2index = {}
        relations = []

        for l in f_anno:
            try:
                stripped_line = l.strip("\n").split("\t")
                if stripped_line[0].startswith('T'):
                    entity = parse_entity(stripped_line, char2word_position)
                    indicator2index[stripped_line[0]] = len(entities)
                    entities.append(entity) 
                else:
                    relation = parse_relation(stripped_line)
                    relations.append(relation)
            except:
                continue

        dataset.append({'tokens':tokens, 'entities':entities, 'relations':relations, 'orig_id':f.replace('.ann', '')})

    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(dataset, out_file)

def merge_json(in1, in2, out):
    with open(in1, 'r', encoding='utf-8') as f1:
        json1 = json.load(f1)

    with open(in2, 'r', encoding='utf-8') as f2:
        json2 = json.load(f2)

    json3 = json1 + json2
    with open(out, 'w', encoding='utf-8') as f3:
        json.dump(json3, f3)

if __name__ == "__main__":
    in1 = 'data/datasets/conll03/conll03_train.json'
    in2 = 'data/datasets/conll03/conll03_dev.json'
    out = 'data/datasets/conll03/conll03_traindev.json'
    merge_json(in1, in2, out)