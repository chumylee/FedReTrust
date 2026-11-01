"""Measure per-block contributions without inventing a scalar client weight."""
from pathlib import Path
import copy,time
import torch
from experiment_core import AggregationResult,file_sha256
from rfa import _flatten_submissions


class BlockContributionRecorder:
    def __init__(self,base):
        self.base,self.name=base,base.name
        self.settings={**copy.deepcopy(base.settings),'measurement':{'name':'block_contribution_recorder_v1',
            'implementation_sha256':file_sha256(Path(__file__)),'global_scalar_coefficient':'undefined',
            'gram_matrix':'sum of actual per-block applied-contribution Gram matrices'}}

    def reset(self):self.base.reset()

    @torch.inference_mode()
    def aggregate(self,submissions,broadcast,round_number):
        start=time.perf_counter();result=self.base.aggregate(submissions,broadcast,round_number)
        defence=time.perf_counter()-start;start=time.perf_counter()
        if result.coefficients or result.diagnostics['coefficient_convention']!='per_block_effective_coefficients_on_raw_client_deltas':
            raise ValueError('Block coefficient contract required.')
        ordered=sorted(submissions,key=lambda s:s.client_id);points=_flatten_submissions(ordered,broadcast)
        positions={};offset=0
        for name,value in broadcast.items():positions[name]=(offset,offset+value.numel());offset+=value.numel()
        contributions=torch.zeros_like(points);seen=set();blocks=copy.deepcopy(result.diagnostics['blocks'])
        for name,block in blocks.items():
            names=block['parameter_names']
            if len(names)!=len(set(names)) or seen.intersection(names) or any(n not in positions for n in names):raise ValueError('Invalid block coverage.')
            seen.update(names);ix=torch.cat([torch.arange(*positions[n]) for n in names])
            coeff=torch.tensor([block['client_metrics'][str(s.client_id)]['effective_coefficient'] for s in ordered],dtype=torch.float64)
            if not torch.isfinite(coeff).all() or (coeff<0).any():raise ValueError('Invalid block coefficients.')
            applied=points[:,ix]*coeff[:,None];contributions[:,ix]=applied
            gram=applied@applied.T
            block['contribution_gram_matrix']=gram.tolist();block['client_order']=[s.client_id for s in ordered]
            block['aggregate_l2_norm']=float(torch.linalg.vector_norm(applied.sum(0)))
            for i,s in enumerate(ordered):
                block['client_metrics'][str(s.client_id)].update({'raw_delta_l2_norm':float(torch.linalg.vector_norm(points[i,ix])),
                    'applied_contribution_l2_norm':float(torch.linalg.vector_norm(applied[i]))})
        if seen!=set(broadcast):raise ValueError('Incomplete block coverage.')
        total=contributions.sum(0);error=0.
        for name,reference in broadcast.items():
            a,b=positions[name];expected=(reference.double()+total[a:b].reshape(reference.shape)).to(reference.dtype)
            torch.testing.assert_close(expected,result.state[name],rtol=2e-6,atol=2e-8)
            error=max(error,float((expected.double()-result.state[name].double()).abs().max()))
        gram=contributions@contributions.T;norms=torch.linalg.vector_norm(contributions,dim=1)
        raw_norms=torch.linalg.vector_norm(points,dim=1);total_norm=float(torch.linalg.vector_norm(total));norm_sum=float(norms.sum())
        metrics={str(s.client_id):{'applied_delta_coefficient':None,'raw_delta_l2_norm':float(raw_norms[i]),
            'applied_contribution_l2_norm':float(norms[i]),
            'contribution_norm_share':float(norms[i])/norm_sum if norm_sum else None,
            'aggregate_projection_share':float(torch.dot(contributions[i],total))/total_norm**2 if total_norm else None}
            for i,s in enumerate(ordered)}
        info={**result.diagnostics,'blocks':blocks,'client_metrics':metrics,
            'contribution_measurements':{'client_order':[s.client_id for s in ordered],'gram_matrix':gram.tolist(),
                'aggregate_l2_norm':total_norm,'sum_individual_norms':norm_sum,'effective_coefficient_mass':None,
                'cancellation_ratio':total_norm/norm_sum if norm_sum else None,
                'reconstruction_max_abs_error':error,'defence_aggregation_seconds':defence,
                'metric_recording_seconds':time.perf_counter()-start}}
        return AggregationResult(result.state,{},info)
