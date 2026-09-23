import csv
import json
import tempfile
import unittest
from pathlib import Path
from pmc.data import prepare, assert_disjoint, verify_artifacts, feature_text
from pmc.labels import supervision, active_labels, decode
import numpy as np
from pmc.metrics import report, tune_threshold
from pmc.hierarchy import label_state

SCHEMA={'subjects':['1','2'],'labels':['10','11','20'], 'nodes':{
 '1':{'path':['1'],'subject':'1','name':'Math'},'2':{'path':['2'],'subject':'2','name':'Physics'},
 '10':{'path':['1','10'],'subject':'1','name':'Parent'},
 '11':{'path':['1','10','11'],'subject':'1','name':'Child'},
 '20':{'path':['2','20'],'subject':'2','name':'Other'}}}

class Core(unittest.TestCase):
 def test_partial(self):
  row={'targets':['10'],'subject':'1'}
  self.assertEqual(supervision(row,SCHEMA),([1,0,0],[1,0,1]))
  self.assertEqual(supervision({'targets':['11']},SCHEMA),([1,1,0],[1,1,1]))
  self.assertEqual(active_labels([row],SCHEMA),[True,False,False])
 def test_subject_and_support(self):
  self.assertEqual(decode([1,1,1],'1',SCHEMA,[True,False,True],[.5]*3),['10'])
 def test_metrics(self):
  rows=[{'targets':['10'],'subject':'1'}]
  probabilities=np.asarray([[.9,.9,.01]])
  result=report(rows,SCHEMA,[0],probabilities,.5,{'10':1})
  self.assertEqual(result['topics_masked']['micro_f1'],1)
  self.assertEqual(result['hierarchical_masked_without_roots']['micro_f1'],1)
  self.assertAlmostEqual(tune_threshold(rows,SCHEMA,probabilities,'validation'),.05)
  with self.assertRaises(ValueError): tune_threshold(rows,SCHEMA,probabilities,'test')
  self.assertEqual(label_state(['11'],SCHEMA),([0,1,0],[0,1,1]))
 def test_feature_boundary(self):
  self.assertEqual(feature_text({'task':'text','answer':'SECRET'}),'text')
 def test_overlap(self):
  rows=[{'id':'1','task':'a','semantic_group_id':'g','split':'train'},
        {'id':'2','task':'b','semantic_group_id':'g','split':'test'}]
  with self.assertRaises(ValueError):assert_disjoint(rows)
 def test_prepare(self):
  with tempfile.TemporaryDirectory() as tmp:
   src=Path(tmp)/'source';src.mkdir();out=Path(tmp)/'out'
   def save(name,fields,rows):
    with (src/name).open('w',newline='') as f:
     writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
   rows=[dict(id=str(i),task=f'text {i}',semantic_group_id=g,split=s,
              target_topic_ids='["11"]',subject_ids='["1"]',unknown_topic_ids='[]')
         for i,g,s in [(1,'a','train'),(2,'b','validation'),(3,'c','test'),(4,'d','train'),(5,'d','train')]]
   save('tasks_normalized_v2.csv',list(rows[0]),rows)
   save('task_topic_links_v2.csv',['topic_path_ids','topic_path_names'],
        [{'topic_path_ids':'["1","10","11"]','topic_path_names':'["Math","Parent","Child"]'}])
   save('manual_review_queue_v2.csv',['task_id','semantic_group_id'],[{'task_id':'4','semantic_group_id':'d'}])
   manifest=prepare(src,out)
   self.assertEqual(manifest['excluded_rows'],2)
   self.assertEqual(manifest['split_counts'],{'train':1,'validation':1,'test':1})
   verify_artifacts(out)
   with self.assertRaises(FileExistsError):prepare(src,out)
   (out/'test.jsonl').write_text('tampered')
   with self.assertRaises(ValueError):verify_artifacts(out)

if __name__=='__main__': unittest.main()
