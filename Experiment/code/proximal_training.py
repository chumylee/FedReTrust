"""FedProx local objective with the project's unchanged SGD/momentum budget.

The simulator can let label-poisoning clients bypass the local penalty. No
attack flags, labels or evaluation data are passed to the aggregator.
"""
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import experiment_core as core

ORIGINAL_TRAIN_CLIENT = core.train_client


def proximal_penalty(model, anchor, mu):
    if isinstance(mu,bool) or not isinstance(mu,(float,int)) or not math.isfinite(mu) or mu<0:
        raise ValueError('mu must be finite and nonnegative.')
    named=dict(model.named_parameters())
    if set(named)!=set(anchor):raise ValueError('Anchor must cover every trainable parameter.')
    values=[]
    for name,param in named.items():
        ref=anchor[name]
        if ref.shape!=param.shape or ref.dtype!=param.dtype or ref.device!=param.device or ref.requires_grad:
            raise ValueError('Detached, matching anchor parameters required.')
        values.append((param-ref).square().sum())
    return (.5*mu)*torch.stack(values).sum()


def effective_mu(mu, attack_policy, label_attack_active):
    if isinstance(mu,bool) or not isinstance(mu,(float,int)) or not math.isfinite(mu) or mu<0:
        raise ValueError('Invalid proximal strength.')
    if attack_policy not in ('compliant','bypass') or type(label_attack_active) is not bool:
        raise ValueError('Invalid simulator attack policy.')
    return 0. if attack_policy=='bypass' and label_attack_active else float(mu)


def train_prox_client(global_model,view,config,client_id,round_number,broadcast_state,mu):
    effective_mu(mu,'compliant',False)
    local_model=copy.deepcopy(global_model)
    if not all(torch.equal(value,broadcast_state[name]) for name,value in local_model.state_dict().items()):
        raise ValueError('Local model must start at the current broadcast.')
    anchor={name:broadcast_state[name].detach().clone() for name,_ in local_model.named_parameters()}
    optimizer=torch.optim.SGD(local_model.parameters(),lr=config.learning_rate,momentum=config.momentum)
    assert len(optimizer.state)==0
    shuffle_seed=(config.development_seed*1_000_003+round_number*1009+client_id)%(2**63-1)
    loader=DataLoader(view,batch_size=config.batch_size,shuffle=True,
                      generator=torch.Generator().manual_seed(shuffle_seed),num_workers=0)
    criterion=nn.CrossEntropyLoss();local_model.train()
    ce_losses,penalties,objectives=[],[],[];first_penalty=None
    for _ in range(config.local_epochs):
        total,penalty_total,objective_total,count=0.,0.,0.,0
        for images,targets in loader:
            optimizer.zero_grad(set_to_none=True)
            ce=criterion(local_model(images),targets)
            # mu=0 takes exactly the original numerical optimization path.
            penalty=proximal_penalty(local_model,anchor,mu) if mu else ce.new_zeros(())
            objective=ce+penalty if mu else ce
            if not torch.isfinite(objective):raise FloatingPointError('Non-finite local objective.')
            if first_penalty is None:first_penalty=float(penalty.detach())
            objective.backward();optimizer.step()
            n=targets.numel();total+=ce.item()*n;penalty_total+=float(penalty.detach())*n
            objective_total+=float(objective.detach())*n;count+=n
        ce_losses.append(total/count);penalties.append(penalty_total/count);objectives.append(objective_total/count)
    state=core.clone_state(local_model)
    if not all(torch.isfinite(value).all() for value in state.values()):raise FloatingPointError('Non-finite local state.')
    if first_penalty!=0.:raise AssertionError('Proximal anchor was not reset to the current broadcast.')
    return state,{'online_train_loss':float(np.mean(ce_losses)),'shuffle_seed':shuffle_seed,
        'optimizer_steps':len(loader)*config.local_epochs,'effective_prox_mu':float(mu),
        'online_proximal_penalty':float(np.mean(penalties)),
        'online_total_objective':float(np.mean(objectives)),'first_step_proximal_penalty':first_penalty}


@contextmanager
def installed_trainer(mu,attack_policy):
    """Process-local injection; preserve the frozen runner file and restore on error.

    view.flip_labels is simulator-only attacker configuration. It controls the
    adversary's choice to ignore a local penalty, never server classification.
    Each worker process executes one experiment; no shared concurrent patch.
    """
    effective_mu(mu,attack_policy,False)
    if core.train_client is not ORIGINAL_TRAIN_CLIENT:raise RuntimeError('Another trainer is already installed.')
    def train(global_model,view,config,client_id,round_number,broadcast_state):
        active=bool(getattr(view,'flip_labels',False))
        applied=effective_mu(mu,attack_policy,active)
        state,metrics=train_prox_client(global_model,view,config,client_id,round_number,broadcast_state,applied)
        metrics['proximal_penalty_bypassed']=bool(active and attack_policy=='bypass' and mu>0)
        return state,metrics
    core.train_client=train
    try:yield
    finally:core.train_client=ORIGINAL_TRAIN_CLIENT


class TrainingTreatment:
    """Label the changed local objective while preserving the aggregator exactly."""
    def __init__(self,base,name,mu):
        effective_mu(mu,'compliant',False)
        self.base,self.name=base,f'{name}_mu_{mu:g}'
        self.settings={**copy.deepcopy(base.settings),'local_training_objective':{
            'name':'FedProx_objective_with_original_SGD_momentum','mu':float(mu),
            'formula':'minibatch_mean_cross_entropy + (mu/2)*sum_trainable_parameters((local-broadcast)^2)',
            'anchor':'current round broadcast, detached and reset for every client call',
            'parameter_count_normalization':False,'optimizer':'original SGD; learning_rate0.01, momentum0.9; fresh per client/round',
            'implementation_sha256':core.file_sha256(Path(__file__))}}
    def reset(self):
        if hasattr(self.base,'reset'):self.base.reset()
    def aggregate(self,submissions,broadcast,round_number):
        return self.base.aggregate(submissions,broadcast,round_number)
