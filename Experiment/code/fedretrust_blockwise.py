"""FedReTrust v0.4: protected classifier with recoverable, bounded feature residuals.

An exploratory combination, not a reproduction of FedCC/centered clipping or
a robustness theorem. Input: submitted models/counts, broadcast, round only.
"""
from dataclasses import asdict,dataclass
from pathlib import Path
import math
import torch
from experiment_core import AggregationResult,file_sha256
from rfa import _flatten_submissions,smoothed_weiszfeld,RFAConfig


@dataclass(frozen=True)
class BlockwiseConfig:
    head_prefix:str='classifier.3'
    feature_radius:float=6.
    norm_cap:float=6.
    restoration_step:float=.5
    epsilon:float=1e-12
    all_block_rfa:bool=False

    def __post_init__(self):
        if not isinstance(self.head_prefix,str) or not self.head_prefix or self.head_prefix.endswith('.'):
            raise ValueError('A nonempty classifier parameter prefix is required.')
        if type(self.all_block_rfa) is not bool:raise ValueError('Boolean control switch required.')
        for name in ('feature_radius','norm_cap','restoration_step','epsilon'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0:
                raise ValueError('Positive finite configuration required: '+name)
        if self.restoration_step>1:raise ValueError('Restoration step must not exceed one.')


def block_layout(broadcast,head_prefix):
    blocks={};offset=0
    for name,value in broadcast.items():
        owner=name.rsplit('.',1)[0] if '.' in name else name
        blocks.setdefault(owner,[]).append((name,offset,offset+value.numel()))
        offset+=value.numel()
    if head_prefix not in blocks or len(blocks)<2:raise ValueError('Require a classifier block and at least one feature block.')
    return blocks


class BlockwiseFedReTrust:
    def __init__(self,options=None):
        self.options=options or BlockwiseConfig()
        self.name='FedReTrust-v04-block-RFA-control' if self.options.all_block_rfa else 'FedReTrust-v04'
        self.settings={**asdict(self.options),'version':'0.4-exploratory',
            'trusted_reference_examples':0,'server_learning_rate':1.,
            'blocks':'weight and bias of each parameter-owner module, jointly flattened',
            'head':'three-step RFA of norm-capped classifier updates; no feature residual correction',
            'features':'RFA center plus sample-weighted, clipped residuals; immediate restriction and capped restoration',
            'coefficient_contract':'per-block effective coefficients; no single global client coefficient exists',
            'initial_memory':'fresh client/block starts at1; no assumed clean warm-up',
            'sample_count_assumption':'authentic counts and stable client identities; no Sybil protection',
            'secure_aggregation_implemented':False,
            'implementation_sha256':file_sha256(Path(__file__)),
            'rfa_implementation_sha256':file_sha256(Path(__file__).with_name('rfa.py'))}
        self.reset()

    def reset(self):
        self.memory={};self.last_round=0;self.layout_signature=None

    @torch.inference_mode()
    def aggregate(self,submissions,broadcast,round_number):
        if type(round_number) is not int or round_number<=self.last_round:raise ValueError('Rounds must increase; reset between runs.')
        if any(type(s.client_id) is not int for s in submissions):raise ValueError('Integer client identities required.')
        ordered=sorted(submissions,key=lambda s:s.client_id)
        points=_flatten_submissions(ordered,broadcast)
        cfg=self.options;blocks=block_layout(broadcast,cfg.head_prefix)
        signature=tuple((n,tuple(t.shape),str(t.dtype)) for n,t in broadcast.items())
        if self.layout_signature is not None and signature!=self.layout_signature:raise ValueError('Model layout changed.')
        base=torch.tensor([s.num_examples for s in ordered],dtype=torch.float64);base/=base.sum()
        delta=torch.zeros_like(points[0]);updated=dict(self.memory);reports={}
        for block,pieces in blocks.items():
            ix=torch.cat([torch.arange(start,end) for _,start,end in pieces])
            raw=points[:,ix];norms=torch.linalg.vector_norm(raw,dim=1)
            cap=cfg.norm_cap*float(norms.median())
            clip=(cap/norms.clamp_min(cfg.epsilon)).clamp(max=1.)
            clip[norms<=cfg.epsilon]=1.
            bounded=raw*clip[:,None]
            center,weights,solver=smoothed_weiszfeld(bounded,base,RFAConfig())
            distances=torch.linalg.vector_norm(bounded-center,dim=1)
            radius=cfg.feature_radius*float(distances.median())
            protected=block==cfg.head_prefix or cfg.all_block_rfa
            if protected:
                evidence=torch.zeros_like(base);q=torch.zeros_like(base)
            else:
                evidence=(radius/distances.clamp_min(cfg.epsilon)).clamp(max=1.)
                evidence[distances<=cfg.epsilon]=1.
                q=torch.tensor([min(float(evidence[i]),self.memory.get((s.client_id,block),1.)+cfg.restoration_step)
                                for i,s in enumerate(ordered)],dtype=torch.float64)
                updated.update({(s.client_id,block):float(q[i]) for i,s in enumerate(ordered)})
            mass=float(torch.dot(base,q))
            # center + sum_i base_i*q_i*(bounded_i-center), expressed on RAW updates.
            coefficients=(base*q+max(0.,1.-mass)*weights)*clip
            block_delta=(coefficients[:,None]*raw).sum(0)
            direct=center+((base*q)[:,None]*(bounded-center)).sum(0)
            torch.testing.assert_close(block_delta,direct,rtol=1e-10,atol=1e-12)
            if not torch.isfinite(block_delta).all():raise FloatingPointError('Non-finite block aggregate.')
            delta[ix]=block_delta
            reports[block]={'parameter_names':[n for n,_,_ in pieces],'protected_head':block==cfg.head_prefix,
                'rfa_only':protected,'norm_cap':cap,'residual_radius':radius,'residual_mass':mass,
                'rfa_iterations':solver['iterations_used'],'client_metrics':{
                    str(s.client_id):{'sample_share':float(base[i]),'norm_clip_factor':float(clip[i]),
                        'bounded_distance_to_center':float(distances[i]),'residual_evidence':float(evidence[i]),
                        'residual_allowance':float(q[i]),'anchor_coefficient':float(weights[i]),
                        'effective_coefficient':float(coefficients[i])} for i,s in enumerate(ordered)}}
        state={};offset=0
        for name,reference in broadcast.items():
            size=reference.numel();state[name]=(reference.double()+delta[offset:offset+size].reshape(reference.shape)).to(reference.dtype)
            if not torch.isfinite(state[name]).all():raise FloatingPointError('Non-finite reconstructed model.')
            offset+=size
        self.memory,self.last_round,self.layout_signature=updated,round_number,signature
        # The frozen runner writes a missing scalar coefficient as a blank CSV cell.
        # Actual contributions are recorded by the dedicated block recorder.
        return AggregationResult(state,{}, {'coefficient_convention':'per_block_effective_coefficients_on_raw_client_deltas',
            'blocks':reports,'reference_forward_examples':0,'single_client_coefficient_defined':False})
