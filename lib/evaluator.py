import warnings
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast
from torch.utils.data import TensorDataset
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')


class Evaluator(object):
    """
    Simple class wrapping the validation procedure
    for a finetuned distilBERT model.
    """
    def __init__(self,
                 model_name,
                 state_dict_path,
                 data_path,
                 out_path,
                 batch_size):
        super(Evaluator, self).__init__()
        self.model_name: str = model_name
        self.state_dict_path: str = state_dict_path
        self.data_path: str = data_path
        self.batch_size: int = batch_size
        self.out_path: str = None
        self.data: pd.DataFrame = None
        self.tfdataset = None
        self.labels = {}
        self.model = None
        self.tokenizer = None
        self.device = None

    def __f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def __accuracy_per_class(self, predictions, true_vals):
        label_dict_inverse = {v: k for k, v in self.labels.items()}
        preds_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = true_vals.flatten()
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

    def __evaluate(self, dataloader):
        self.model.eval()
        loss_val_total = 0
        predictions, true_vals = [], []
        for batch in dataloader:
            batch = tuple(b.to(self.device) for b in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                      }
            with torch.no_grad():
                outputs = self.model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        loss_val_avg = loss_val_total/len(dataloader)
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        return loss_val_avg, predictions, true_vals

    def __get_assets(self):
        self.data = pd.read_csv(self.data_path,
                                names=['text', 'text_label'])
        possible_labels = self.data.text_label.unique()
        for index, possible_label in enumerate(possible_labels):
            self.labels[possible_label] = index
        self.data['label'] = self.data.text_label.replace(self.labels)
        self.tokenizer = (DistilBertTokenizerFast
                          .from_pretrained(self.tokenizer_name,
                                           do_lower_case=True))
        self.model = (DistilBertForSequenceClassification.
                      from_pretrained(self.model_name,
                                      num_labels=len(self.labels),
                                      output_attentions=False,
                                      output_hidden_states=False))
        self.model.load_state_dict(
            torch.load(self.state_dict_path,
                       map_location=torch.device('cpu')))

    def __prepare_data(self):
        encoded_data = self.tokenizer(
            self.data.text.to_list(),
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt')

        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = (torch.tensor(self.data.label.values))

        self.tfdataset = TensorDataset(input_ids,
                                       attention_masks,
                                       labels)

    def __get_predictions(self, predictions, true_vals):
        output = pd.DataFrame({'predictions': predictions,
                               'true_vals': true_vals})
        output.to_csv(self.out_path, index=False)

    def run(self, save_predictions=False):
        print('Collecting assets... ', end='', flush=True)
        self.__get_assets()
        print('[DONE]')

        print('Prepareing data... ', end='', flush=True)
        self.__prepare_data()
        print('[DONE]')

        dataloader = DataLoader(
            self.tfdataset,
            sampler=SequentialSampler(self.tfdataset),
            batch_size=self.batch_size)
        self.device = (torch.device('cuda' if
                                    torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        print(f'Ready to evaluate on: {self.device}')

        _, predictions, true_vals = self.__evaluate(dataloader)

        self.__accuracy_per_class(predictions, true_vals)
        print('F1 Score:')
        print(self.__f1_score_func(predictions, true_vals))

        if save_predictions:
            if self.out_path:
                self.__get_predictions(predictions, true_vals)
