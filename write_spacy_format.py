import json
from spacy.lang.en import English   # or whichever language tokenizer you need
import pprint
import hashlib
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer

pp = pprint.PrettyPrinter(indent=4)

nlp = English()
id_token = 0
files = ['data/train/20200310-110200/dataset.json', 'data/test/20200310-104041/dataset.json']
# JSON_FILE = 'data/datasets/dataset.json'
SAVE_FILE = ['spacy_ds_train.jsonl', 'spacy_ds_test.jsonl']
import nltk
nltk.download('stopwords')

# lim = 100
# for x, name in enumerate(files):
#     with open(name, 'r') as jf:
#         spert_data = json.load(jf)
#     fin_docs = []
#     documents = {}
#     for y, sentence in enumerate(spert_data):
#         if y > lim:
#             break
#         if sentence['orig_id'] not in documents:
#             documents[sentence['orig_id']] = {"raw": '',
#                                     "sentences": [
#                                     ]}
#         txt = TreebankWordDetokenizer().detokenize(sentence['tokens'])
#         spans = [span for span in TreebankWordTokenizer().span_tokenize(txt)]
#         tokens_real = [token for token in TreebankWordTokenizer().tokenize(txt)]
#         tokens = []
#         for id_token, word in enumerate(tokens_real):
#             tokens.append({
#                            "tag": None,
#                            "orth": word,
#                            "ner": "O"
#                            })
#             id_token += 1
#         for ent in sentence['entities']:
#             for i in range(ent['start'], ent['end']):
#                 tokens[i]["ner"] = ent["type"]
#
#         documents[sentence['orig_id']]["sentences"].append({"tokens": tokens,
#                                                             "brackets": []
#                                                             })
#
#         documents[sentence['orig_id']]["raw"] += TreebankWordDetokenizer().detokenize(sentence['tokens'])
#     for document, value in documents.items():
#         doc_dict = {
#             "id": document,
#             "paragraphs": [{
#                 # "raw": value["raw"],
#                 "sentences": value["sentences"]
#
#         }]
#         }
#         fin_docs.append(doc_dict)
#     with open(SAVE_FILE[x], 'w') as jf:
#         json.dump(fin_docs, jf)
#         # for line in fin_docs:
#         #     json.dump(line, jf)
#         #     jf.write('\n')
#     # print(documents[sentence['orig_id']]["raw"])

for i, name in enumerate(files):
    with open(name, 'r') as jf:
        spert_data = json.load(jf)
    final_documents = []

    for sentence in spert_data:
        sent_dict = {}
        sent_dict["text"] = TreebankWordDetokenizer().detokenize(sentence['tokens'])
        spans = [span for span in TreebankWordTokenizer().span_tokenize(sent_dict["text"])]
        tokens = [token for token in TreebankWordTokenizer().tokenize(sent_dict["text"])]
        # sent_dict["meta"] = {"section": sentence['orig_id']}
        # sent_dict["_input_hash"] = hashlib.md5(sent_dict["text"].encode()).hexdigest()
        # sent_dict["_task_hash"] = hashlib.md5(sentence['orig_id'].encode()).hexdigest()
        toks = [{"text": token,
                                "start": spans[token_id][0],
                                "end":spans[token_id][1],
                                "id": token_id} for token_id, token in enumerate(tokens)]
        # print(sentence)
        # for entity in sentence['entities']:
        #     print(sent_dict["tokens"][entity["start"]]["start"])
        #     # print(sent_dict["tokens"][entity["end"]]["end"])
        #     print(len(sent_dict["tokens"]))
        #     print(entity["end"])
        sent_dict["entities"] = sorted([(toks[entity["start"]]["start"], toks[entity["end"]-1]["end"], entity["type"])
                    #   "start": sent_dict["tokens"][entity["start"]-1]["start"],
                    #   "end": sent_dict["tokens"][entity["end"]-1]["end"],
                    #   # "token_start": entity["start"],
                    #   # "token_end": entity["end"],
                    #   "label": entity["type"]
                    # }
            for entity in sentence['entities']], key=lambda tup: tup[0])

        for ent1 in sent_dict["entities"][:-1]:
            for ent2 in sent_dict["entities"][1:]:
                if ent1[1] > ent2[0]:
                    print(f'Removing: {ent1} and {ent2}')

                    if ent1 in sent_dict["entities"] and ent2 in sent_dict["entities"]:
                        sent_dict["entities"].append((ent1[0], ent2[1], random.choice([ent1[2], ent2[2]])))
                        sent_dict["entities"].remove(ent1)
                        sent_dict["entities"].remove(ent2)
        for ent1 in sent_dict["entities"][:-1]:
            for ent2 in sent_dict["entities"][1:]:
                if ent1[1] > ent2[0]:
                    print(f'Removing: {ent1} and {ent2}')

                    if ent1 in sent_dict["entities"] and ent2 in sent_dict["entities"]:
                        sent_dict["entities"].append((ent1[0], ent2[1], random.choice([ent1[2], ent2[2]])))
                        sent_dict["entities"].remove(ent1)
                        sent_dict["entities"].remove(ent2)

        # sent_dict["_session_id"] = None
        # sent_dict["_view_id"] = "ner_manual"
        # sent_dict["answer"] = "accept"
        final_documents.append(sent_dict)
    fin_docs = [(value["text"], {"entities": value["entities"]}) for value in final_documents]
    with open(SAVE_FILE[i], 'w') as jf:
        json.dump(fin_docs, jf)
        # for line in final_documents:
        #     json.dump(line, jf)
        #     jf.write('\n')

