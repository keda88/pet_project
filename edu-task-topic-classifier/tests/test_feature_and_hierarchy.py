"""Stdlib checks for the API delivered by the resumed initial task."""
import ast
import tempfile
import unittest
from pathlib import Path
from pmc import data
from pmc.cli import reserve_test
from pmc.hierarchy import label_state, close_topics, most_specific

SCHEMA = {'labels':['2','3','4','5'],'subjects':['1'],'nodes':{
    t:{'path':p,'name':t,'subject':'1'} for t,p in {
        '1':['1'],'2':['1','2'],'3':['1','2','3'],'4':['1','2','4'],'5':['1','5']}.items()}}


class SafetyTests(unittest.TestCase):
    def test_feature_invariance(self):
        row={'task':'Условие', 'answer':'SECRET', 'text_of_solution':'SECRET',
             'id':'99','split':'test','topic_paths':'SECRET','quality_flags':'SECRET'}
        for key in row:
            if key != 'task':
                self.assertEqual(data.feature_text(dict(row,**{key:'injected label'})),'Условие')

    def test_feature_allowlist_structure(self):
        tree=ast.parse(Path(data.__file__).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='feature_text')
        self.assertEqual([n.slice.value for n in ast.walk(fn) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)],['task'])

    def test_empty_task(self):
        with self.assertRaises(ValueError): data.feature_text({'task':' '})

    def test_group_overlap(self):
        rows=[dict(id='1',task='А',semantic_group_id='g',split='train'),dict(id='2',task='Б',semantic_group_id='g',split='test')]
        with self.assertRaises(ValueError): data.assert_disjoint(rows)

    def test_exact_overlap(self):
        rows=[dict(id='1',task='А  Б',semantic_group_id='g1',split='train'),dict(id='2',task='А Б',semantic_group_id='g2',split='validation')]
        with self.assertRaises(ValueError): data.assert_disjoint(rows)

    def test_same_split(self):
        data.assert_disjoint([dict(id=str(i),task='А',semantic_group_id='g',split='train') for i in range(2)])

    def test_intermediate_descendants_unknown(self):
        self.assertEqual(label_state(['2'],SCHEMA),([1,0,0,0],[1,0,0,1]))

    def test_leaf_ancestor_unknown(self):
        self.assertEqual(label_state(['3'],SCHEMA),([0,1,0,0],[0,1,1,1]))

    def test_positive_overrides_unknown(self):
        self.assertEqual(label_state(['2','3'],SCHEMA),([1,1,0,0],[1,1,0,1]))

    def test_hierarchical_closure(self):
        self.assertEqual(label_state(['3'],SCHEMA,hierarchical=True),([1,1,0,0],[1,1,1,1]))
        self.assertEqual(close_topics(['3','5'],SCHEMA),{'1','2','3','5'})
        self.assertEqual(most_specific(['2','3','5'],SCHEMA),['3','5'])

    def test_root_only(self):
        self.assertEqual(label_state(['1'],SCHEMA),([0,0,0,0],[0,0,0,0]))

    def test_test_once(self):
        with tempfile.TemporaryDirectory() as directory:
            reserve_test(directory,'run1')
            with self.assertRaises(FileExistsError): reserve_test(directory,'run2')


if __name__=='__main__': unittest.main()
