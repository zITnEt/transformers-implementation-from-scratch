import nltk
import json

with open("data/raw/rus.txt", encoding="utf-8") as f:
    src_dict = {"<pad>": 0, "<sos>":1, "<eos>": 2, "<unk>": 3}
    trg_dict = {"<pad>": 0, "<sos>":1, "<eos>": 2, "<unk>": 3}
    src_id=4
    trg_id=4

    for line in f:
        ar = line.strip().split('\t')
        if len(ar) < 2:
            continue
        src = ar[0]
        trg = ar[1]

        src_tokens = nltk.word_tokenize(src.lower(), language='english')
        trg_tokens = nltk.word_tokenize(trg.lower(), language='russian')

        for tok in src_tokens:
            if tok not in src_dict:
                src_dict[tok]=src_id
                src_id+=1
        
        for tok in trg_tokens:
            if tok not in trg_dict:
                trg_dict[tok]=trg_id
                trg_id+=1

with open("tokenizer/vocab_en.json", "w", encoding="utf-8") as f:
    json.dump(src_dict, f, ensure_ascii=False, indent=2) 

with open("tokenizer/vocab_ru.json", "w", encoding="utf-8") as f:
    json.dump(trg_dict, f, ensure_ascii=False, indent=2)