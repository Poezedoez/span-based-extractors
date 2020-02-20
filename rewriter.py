import json

def rewrite_conll03(input_path, output_path):
    dataset = []
    with open(input_path, 'r', encoding='utf-8') as in_file:
        id_ = 0
        tokens = []
        entities = []
        relations = []
        entity = {'type':'O', 'start':0, 'end':0}
        previous_label = 'O'

        for line in in_file:
            ## No data yet
            if line == '\n' and not tokens:
                continue

            ## Empty line means end of sentence
            if line == '\n':
                dataset.append({'tokens':tokens, 'entities':entities, 'relations':relations, 'orig_id':id_})
                id_ += 1
                tokens = []
                entities = []
                continue

            token, _, _, label = line.strip().split(" ")
            
            ## End of previous entity span
            if label != previous_label:
                entity['end'] = len(tokens)
                ## Don't add O-type
                if entity['type'] != 'O':
                    entities.append(entity)
                ## Create new entity span
                type_ = label.split('-')[1] if label != 'O' else 'O'
                entity = {'type':type_, 'start':len(tokens)}

            tokens.append(token)
            previous_label = label

    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(dataset, out_file)

if __name__ == "__main__":
    input_path = 'data/datasets/conll03/eng.train.txt'
    output_path = 'data/datasets/conll03/conll03_train.json'
    rewrite_conll03(input_path, output_path)