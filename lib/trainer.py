import warnings
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, RobertaTokenizerFast
from torch.utils.data import TensorDataset
from transformers import DistilBertForSequenceClassification, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')


class Trainer(object):
    """
    Simple class wrapper for pyTorch distillBERT
    training process.
    """
    def __init__(self,
                 model_arch,
                 model_name,
                 data_path,
                 out_path,
                 epochs=3,
                 batch_size=8,
                 test_size=0.25):
        super(Trainer, self).__init__()
        self.model_arch: str = model_arch
        self.model_name: str = model_name
        self.data_path: str = data_path
        self.out_path: str = out_path
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.test_size: float = test_size
        self.data: pd.DataFrame = None
        self.labels = {}
        self.model = None
        self.tokenizer = None
        self.dataset_train = None
        self.dataset_val = None
        self.device = None

    def __f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def __evaluate(self, dataloader_val):
        self.model.eval()
        loss_val_total = 0
        predictions, true_vals = [], []
        for batch in dataloader_val:
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
        loss_val_avg = loss_val_total/len(dataloader_val)
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        return loss_val_avg, predictions, true_vals

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

    def __get_data(self):
        self.data = pd.read_csv(self.data_path,
                                names=['text', 'text_label'])
        self.data.dropna(inplace=True)
        possible_labels = self.data.text_label.unique()
        for index, possible_label in enumerate(possible_labels):
            self.labels[possible_label] = index
        self.data['label'] = self.data.text_label.replace(self.labels)

    def __prepare_data(self):
        (X_train, X_val,
         y_train, y_val) = train_test_split(self.data.index.values,
                                            self.data.label.values,
                                            test_size=self.test_size,
                                            stratify=self.data.label.values)
        self.data['data_type'] = ['not_set']*self.data.shape[0]
        self.data.loc[X_train, 'data_type'] = 'train'
        self.data.loc[X_val, 'data_type'] = 'val'

        encoded_data_train = self.tokenizer(
            self.data[self.data.data_type == 'train'].text.to_list(),
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt')

        encoded_data_val = self.tokenizer(
            self.data[self.data.data_type == 'val'].text.to_list(),
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt')

        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = (torch.tensor(self.data[self.data.data_type == 'train']
                                     .label.values))

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = (torch.tensor(self.data[self.data.data_type == 'val']
                                   .label.values))

        self.dataset_train = TensorDataset(input_ids_train,
                                           attention_masks_train,
                                           labels_train)
        self.dataset_val = TensorDataset(input_ids_val,
                                         attention_masks_val,
                                         labels_val)

    def __train(self):
        dataloader_train = DataLoader(
            self.dataset_train,
            sampler=RandomSampler(self.dataset_train),
            batch_size=self.batch_size
        )
        dataloader_validation = DataLoader(
            self.dataset_val,
            sampler=SequentialSampler(self.dataset_val),
            batch_size=self.batch_size
        )
        optimizer = AdamW(
            self.model.parameters(),
            lr=1e-5,
            eps=1e-8
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader_train)*self.epochs
        )
        self.device = (torch.device('cuda' if
                                    torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        print(f'Training on: {self.device}')

        # Main traning loop
        for epoch in tqdm(range(1, self.epochs+1)):
            self.model.train()
            loss_train_total = 0
            progress_bar = tqdm(dataloader_train,
                                desc='Epoch {:1d}'.format(epoch),
                                leave=False,
                                disable=False)
            for batch in progress_bar:
                self.model.zero_grad()
                batch = tuple(b.to(self.device) for b in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[2],
                          }
                outputs = self.model(**inputs)
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                (progress_bar.set_postfix({'training_loss': '{:.3f}'
                                           .format(loss.item()/len(batch))}))

            torch.save(self.model.state_dict(),
                       f'{self.out_path}_epoch_{epoch}.model')
            tqdm.write(f'\nEpoch {epoch}')
            loss_train_avg = loss_train_total/len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')
            (val_loss, predictions,
             true_vals) = self.__evaluate(dataloader_validation)
            val_f1 = self.__f1_score_func(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (Weighted): {val_f1}')

    def run(self):
        print('Loading data... ', end='', flush=True)
        self.__get_data()
        print('[DONE]')
        
        print('Collecting assets... ', end='', flush=True)
        self.__get_model_specific_assets()
        print('[DONE]')

        print('Preparing data... ', end='', flush=True)
        self.__prepare_data()
        print('[DONE]')

        print('Training...')
        self.__train()
        print('Final labels:')
        print(self.labels)
        print('[DONE]')
