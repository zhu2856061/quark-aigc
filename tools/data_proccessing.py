import spacy
from spacy.lang.fr.examples import sentences
import json
from loguru import logger
import time


class DataProccessing(object):

    def __init__(self, source, target) -> None:
        self.source = source
        self.target = target
        self.source_word2index = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }
        self.target_word2index = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }

        if self.source == "fr":
            self.source_nlp = spacy.load("fr_core_news_sm")

        if self.target == "en":
            self.target_nlp = spacy.load("en_core_web_sm")

    def _read_file(self, file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            data_dict = json.load(f)
        return data_dict

    def _token(self, row, tag="source"):
        if tag == "source":
            doc = self.source_nlp(row)
            for word in doc:
                if word.text not in self.source_word2index:
                    self.source_word2index[word.text] = len(self.source_word2index)
            token = ["<SOS>"] + [_.text for _ in doc] + ["<EOS>"]
            return token
        if tag == "target":
            doc = self.target_nlp(row)
            for word in doc:
                if word.text not in self.target_word2index:
                    self.target_word2index[word.text] = len(self.target_word2index)
            token = ["<SOS>"] + [_.text for _ in doc] + ["<EOS>"]
            return token

    def _to_feature(self, source_tokens, target_tokens):
        # self.source_word_len = max([len(_) for _ in source_tokens])
        # self.target_word_len = max([len(_) for _ in target_tokens])

        samples = []
        for i in range(len(source_tokens)):
            tmp_dic = {
                "logTime": int(time.time()),
                # "y": {},
                "x": {},
                "extraInfo": {
                    "tag": "merlin"
                }
            }
            tmp_dic["x"] = {
                "source": {
                    "SVA": source_tokens[i]
                },
                "target": {
                    "SVA": target_tokens[i]
                },
            }
            samples.append(tmp_dic)
        return samples

    def save_vocab(self, dir):
        with open(dir + "/source_vocab", 'w') as f:
            tmp = sorted(self.source_word2index.items(), key=lambda _: _[1])
            for t, _ in tmp:
                f.write(t + "\n")

        with open(dir + "/target_vocab", 'w') as f:
            tmp = sorted(self.target_word2index.items(), key=lambda _: _[1])
            for t, _ in tmp:
                f.write(t + "\n")

    def save_sample(self, samples, file):
        with open(file, 'w') as f:
            for sa in samples:
                f.write(json.dumps(sa) + "\n")

    def proccessing(self, file_path, training=True):

        logger.info(">>>>> read train file: {file_path} <<<<<",
                    file_path=file_path)
        data_dict = self._read_file(file_path)
        source_tokens = []
        target_tokens = []

        logger.info(">>>>> proccessing: {source} -> {target} <<<<<",
                    source=self.source,
                    target=self.target)
        for row in data_dict['rows']:
            source_sentence = row['row']['translation'][self.source]
            source_tokens.append(self._token(source_sentence, "source"))
            target_sentence = row['row']['translation'][self.target]
            target_tokens.append(self._token(target_sentence, "target"))

        samples = self._to_feature(source_tokens, target_tokens)
        self.save_sample(samples, file_path + ".format")
        logger.info(">>>>> proccessing over <<<<<")


if __name__ == '__main__':
    dp = DataProccessing("fr", "en")
    dp.proccessing("../data/wmt14-fr-en-train.json", True)

    dp.proccessing("../data/wmt14-fr-en-valid.json", False)

    dp.proccessing("../data/wmt14-fr-en-test.json", False)

    dp.save_vocab("../data/vocab")
