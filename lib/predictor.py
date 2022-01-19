import warnings
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, RobertaTokenizerFast
from torch.utils.data import TensorDataset
from transformers import DistilBertForSequenceClassification, RobertaForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler

warnings.filterwarnings('ignore')


class Predictor(object):
    """
    Simple class wrapping the inference procedure
    for a finetuned distilBERT model.
    """
    def __init__(self,
                 model_arch,
                 model_name,
                 state_dict_path,
                 labels,
                 text_col,
                 data_path,
                 out_path,
                 batch_size):
        super(Predictor, self).__init__()
        self.model_arch: str = model_arch
        self.model_name: str = model_name
        self.state_dict_path: str = state_dict_path
        self.data_path: str = data_path
        self.out_path: str = out_path
        self.batch_size: int = batch_size
        self.labels: dict = labels
        self.text_col: int = text_col
        self.model = None
        self.tokenizer = None
        self.device = None

    def __inference(self, dataloader, header):
        self.model.eval()
        predictions = []
        ids = []
        for batch in dataloader:
            batch = tuple(item.to(self.device) for item in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

            with torch.inference_mode():
                outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            probs = probs.detach().cpu().numpy()
            predictions.append(probs)
            ids.append(batch[2].detach().cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        ids = np.concatenate(ids, axis=0)
        probs_df = pd.DataFrame(predictions)
        inv_labels = {v: k for k, v in self.labels.items()}
        probs_df.rename(columns=inv_labels, inplace=True)
        ids_df = pd.DataFrame({'id': ids})
        out = pd.concat([ids_df, probs_df], axis=1)
        out.to_csv(self.out_path,
                   mode='a',
                   index=False,
                   header=header)

    def __get_model_specific_assets(self):
        if self.model_arch == 'distilbert':
            self.tokenizer = (DistilBertTokenizerFast
                              .from_pretrained(self.model_name,
                                               do_lower_case=True))
            self.model = (DistilBertForSequenceClassification.
                          from_pretrained(self.model_name,
                                          num_labels=len(self.labels),
                                          output_attentions=False,
                                          output_hidden_states=False))

        elif self.model_arch == 'roberta':
            self.tokenizer = (RobertaTokenizerFast
                              .from_pretrained(self.model_name,
                                               do_lower_case=True))
            self.model = (RobertaForSequenceClassification.
                          from_pretrained(self.model_name,
                                          num_labels=len(self.labels),
                                          output_attentions=False,
                                          output_hidden_states=False))

        self.model.load_state_dict(
            torch.load(self.state_dict_path,
                       map_location=torch.device('cpu')))


    def __prepare_data(self, chunk):
        encoded_data = self.tokenizer(
            chunk.iloc[:, 0].to_list(),
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt')
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        ids = (torch.tensor(chunk.index.values))
        tf_dataset = TensorDataset(input_ids, attention_masks, ids)
        dataloader = DataLoader(tf_dataset,
                                sampler=SequentialSampler(tf_dataset),
                                batch_size=self.batch_size)
        return dataloader

    def run(self):
        print('Collecting assets... ', file=sys.stderr)
        self.__get_model_specific_assets()
        print('[DONE]', file=sys.stderr)

        self.device = (torch.device('cuda' if
                                    torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        with pd.read_csv(self.data_path,
                         usecols=[self.text_col],
                         lineterminator='\n',
                         chunksize=1000000,
                         dtype=str) as reader:
            header = True
            for chunk in reader:
                chunk.dropna(inplace=True)
                mydataloader = self.__prepare_data(chunk)
                self.__inference(mydataloader, header=header)
                header = False
