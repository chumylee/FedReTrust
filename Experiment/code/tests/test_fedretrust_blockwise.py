import copy
from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission
from fedretrust_blockwise import BlockwiseConfig,BlockwiseFedReTrust
from blockwise_recording import BlockContributionRecorder
from rfa import smoothed_weiszfeld


def fixture(feature,head=None):
    if head is None:head=feature
    b={'feature.weight':torch.zeros(2,dtype=torch.float64),'feature.bias':torch.zeros(1,dtype=torch.float64),
       'head.weight':torch.zeros(2,dtype=torch.float64),'head.bias':torch.zeros(1,dtype=torch.float64)}
    s=[ClientSubmission(i,10,{'feature.weight':torch.tensor(f,dtype=torch.float64),'feature.bias':torch.tensor([f[0]/10],dtype=torch.float64),
       'head.weight':torch.tensor(h,dtype=torch.float64),'head.bias':torch.tensor([h[0]/10],dtype=torch.float64)})
       for i,(f,h) in enumerate(zip(feature,head))]
    return b,s


def method(**kw):return BlockwiseFedReTrust(BlockwiseConfig(head_prefix='head',**kw))


class BlockwiseTests(unittest.TestCase):
    def test_head_is_rfa_feature_can_preserve_mean(self):
        points=[[1.,0.],[1.1,.1],[.9,-.1],[2.,.3]]
        b,s=fixture(points);r=BlockContributionRecorder(method(feature_radius=100.)).aggregate(s,b,1)
        raw=torch.stack([torch.cat([x.state['head.weight'],x.state['head.bias']]) for x in s])
        center,_,_=smoothed_weiszfeld(raw,[1.]*4)
        torch.testing.assert_close(torch.cat([r.state['head.weight'],r.state['head.bias']]),center)
        torch.testing.assert_close(r.state['feature.weight'],torch.tensor(points,dtype=torch.float64).mean(0))
        self.assertEqual(r.coefficients,{})
        self.assertIsNone(r.diagnostics['client_metrics']['0']['applied_delta_coefficient'])
        self.assertEqual(r.diagnostics['reference_forward_examples'],0)

    def test_block_control_removes_only_residual_correction(self):
        b,s=fixture([[1.,0.],[1.1,.1],[.9,-.1],[2.,.3]])
        a=BlockContributionRecorder(method(feature_radius=100.)).aggregate(s,b,1)
        c=BlockContributionRecorder(method(feature_radius=100.,all_block_rfa=True)).aggregate(s,b,1)
        torch.testing.assert_close(a.state['head.weight'],c.state['head.weight'])
        self.assertFalse(torch.allclose(a.state['feature.weight'],c.state['feature.weight']))
        self.assertTrue(all(v['residual_mass']==0 for v in c.diagnostics['blocks'].values()))

    def test_group_gram_reconstructs_applied_vector_and_blocks(self):
        b,s=fixture([[1.,0.],[.8,.2],[.9,-.2],[100.,-90.]])
        r=BlockContributionRecorder(method()).aggregate(s,b,1)
        gram=torch.tensor(r.diagnostics['contribution_measurements']['gram_matrix'],dtype=torch.float64)
        block_gram=sum(torch.tensor(v['contribution_gram_matrix'],dtype=torch.float64) for v in r.diagnostics['blocks'].values())
        torch.testing.assert_close(gram,block_gram)
        actual=sum(t.square().sum() for t in r.state.values())
        torch.testing.assert_close(gram.sum(),actual)
        self.assertLess(r.diagnostics['contribution_measurements']['reconstruction_max_abs_error'],1e-12)

    def test_every_block_bounds_large_updates(self):
        for magnitude in (1e3,1e9):
            b,s=fixture([[1.,0.]]*7+[[magnitude,-magnitude]]*3)
            r=BlockContributionRecorder(method()).aggregate(s,b,1)
            for block in r.diagnostics['blocks'].values():
                self.assertLessEqual(block['aggregate_l2_norm'],block['norm_cap']+1e-9)
                self.assertGreaterEqual(block['client_metrics']['9']['effective_coefficient'],0.)

    def test_restriction_repair_absence_reset(self):
        a=method();b,s=fixture([[1.,0.]]*7+[[-100.,0.]]*3)
        a.aggregate(s,b,1);low=a.memory[9,'feature'];self.assertLess(low,.5)
        _,normal=fixture([[1.,0.]]*10)
        a.aggregate(normal[:7],b,2);self.assertEqual(low,a.memory[9,'feature'])
        a.aggregate(normal,b,3);self.assertAlmostEqual(a.memory[9,'feature'],low+.5)
        a.aggregate(normal,b,4);self.assertEqual(a.memory[9,'feature'],1.)
        self.assertNotIn((9,'head'),a.memory)
        a.reset();self.assertEqual(a.memory,{});self.assertIsNone(a.layout_signature);self.assertEqual(a.last_round,0)

    def test_identical_and_zero_updates_and_unequal_counts(self):
        for point in ([0.,0.],[1.,2.]):
            b,s=fixture([point]*4);s[0].num_examples=30
            r=BlockContributionRecorder(method()).aggregate(s,b,1)
            for name in b:torch.testing.assert_close(r.state[name],s[0].state[name])
            for block in r.diagnostics['blocks'].values():self.assertAlmostEqual(block['client_metrics']['0']['effective_coefficient'],.5)

    def test_permutation_translation_input_immutability(self):
        b,s=fixture([[1.,.1],[.8,-.1],[1.1,0.],[5.,-.3]])
        saved=copy.deepcopy(s);rng=torch.random.get_rng_state().clone()
        a=BlockContributionRecorder(method()).aggregate(s,b,1)
        p=BlockContributionRecorder(method()).aggregate(list(reversed(s)),b,1)
        shift={n:torch.full_like(t,3.) for n,t in b.items()}
        moved=[ClientSubmission(x.client_id,x.num_examples,{n:t+shift[n] for n,t in x.state.items()}) for x in s]
        z=BlockContributionRecorder(method()).aggregate(moved,shift,1)
        for n in b:
            torch.testing.assert_close(a.state[n],p.state[n]);torch.testing.assert_close(a.state[n]+shift[n],z.state[n])
        for l,r in zip(s,saved):
            for n in b:torch.testing.assert_close(l.state[n],r.state[n])
        torch.testing.assert_close(torch.random.get_rng_state(),rng)

    def test_invalid_input_does_not_commit_memory(self):
        b,s=fixture([[1.,0.]]*4);a=method();a.aggregate(s,b,1)
        mem=copy.deepcopy(a.memory);layout=a.layout_signature
        for kind in ('nan','duplicate','shape','count'):
            bad=copy.deepcopy(s)
            if kind=='nan':bad[0].state['head.weight'][0]=float('nan')
            elif kind=='duplicate':bad[0].client_id=bad[1].client_id
            elif kind=='shape':bad[0].state['head.weight']=torch.ones(3,dtype=torch.float64)
            else:bad[0].num_examples=0
            with self.assertRaises((ValueError,FloatingPointError)):a.aggregate(bad,b,2)
            self.assertEqual(a.memory,mem);self.assertEqual(a.layout_signature,layout);self.assertEqual(a.last_round,1)
        with self.assertRaises(ValueError):a.aggregate(s,b,1)

    def test_layout_mismatch_and_config_validation(self):
        for kw in ({'feature_radius':0},{'feature_radius':True},{'norm_cap':float('inf')},{'epsilon':0},
                   {'head_prefix':''},{'all_block_rfa':1},{'restoration_step':1.1}):
            with self.assertRaises(ValueError):BlockwiseConfig(**kw)
        b,s=fixture([[1.,0.]]*4)
        with self.assertRaises(ValueError):BlockwiseFedReTrust().aggregate(s,b,1)


if __name__=='__main__':unittest.main()
