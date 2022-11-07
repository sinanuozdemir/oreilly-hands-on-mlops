import logging
from ts.torch_handler.base_handler import BaseHandler
from transformers import DistilBertTokenizerFast
import torch

class TextHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def preprocess(self, requests):
        print(f"preprocess for requests={requests}")
        texts = [r['body'] for r in requests]
        return self.tokenizer.batch_encode_plus(texts, return_tensors='pt')

    def inference(self, x):
        '''
        Perform the model inference
        '''
        print(f"inference for x={x}")
        return self.model(**x)

    def postprocess(self, preds):
        '''
        Torchserve always expects an array to be returned.
        '''
        print(f"postprocess for preds={preds}")
        post = torch.nn.Softmax(dim=1)(preds['logits']).max(1)
        return [
        {'probability': c, 'label': self.mapping[str(label)]} \
            for label, c in zip(post.indices.tolist(), post.values.tolist())
        ]
