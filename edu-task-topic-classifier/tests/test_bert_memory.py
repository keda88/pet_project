import importlib.util
import unittest
from pmc.data import validate_config


class MemoryConfig(unittest.TestCase):
    def test_reject_unsafe_fraction(self):
        for fraction in (0, -1, 1.7, float('nan')):
            with self.assertRaises(ValueError):
                validate_config(dict(kind='bert', feature_fields=['task'], seed=42,
                                     mps_memory_fraction=fraction))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch dependency')
class Inference(unittest.TestCase):
    def test_fixed_shapes(self):
        import torch
        from pmc.bert import batch_tokens
        def tokenizer(texts, **kwargs):
            self.assertEqual(kwargs['padding'], 'max_length')
            return {'input_ids': torch.ones(len(texts), kwargs['max_length'])}
        for text in ('x', 'longer text'):
            tokens = batch_tokens(tokenizer,[text],torch.device('cpu'),128,True)
            self.assertEqual(tuple(tokens['input_ids'].shape),(1,128))

    def test_nested_config_limits(self):
        from pathlib import Path
        import json
        config=json.loads(Path('configs/bert_mac_m4_48gb.json').read_text())
        validate_config(config)
        config['bert']['mps_memory_fraction']=0
        with self.assertRaises(ValueError): validate_config(config)

    def test_predict_disables_gradients(self):
        import torch
        import numpy as np
        from unittest.mock import patch
        from pmc import bert
        observed=[]
        class Model(torch.nn.Module):
            def forward(self, tokens):
                observed.append((torch.is_grad_enabled(),self.training))
                return torch.zeros(1,2),torch.zeros(1,3)
        def tokenizer(texts, **kwargs):
            return {'input_ids':torch.ones(len(texts),kwargs['max_length'])}
        documents={'config.json':{'bert':{'device':'cpu','batch_size':1,'max_length':128,'fixed_padding':True}},
                   'schema.json':{'subjects':['1','2'],'labels':['10','11','20']},'active.json':[True,False,True]}
        with patch.object(bert,'read_json',side_effect=lambda p:documents[p.name]), \
             patch.object(bert.AutoTokenizer,'from_pretrained',return_value=tokenizer), \
             patch.object(bert.AutoConfig,'from_pretrained'), \
             patch.object(bert.AutoModel,'from_config'), \
             patch.object(bert,'DualHead',return_value=Model()), \
             patch.object(torch,'load',return_value={}):
            subjects, probabilities=bert.predict(['short','longer text'],'unused')
        self.assertEqual(observed,[(False,False),(False,False)])
        np.testing.assert_array_equal(subjects,[0,0])
        np.testing.assert_allclose(probabilities,[[.5,0,.5],[.5,0,.5]])
