"""Frozen four-way training comparison; no edits to earlier experimental sources."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import argparse,fcntl,json,platform,resource,shutil,subprocess,sys,time
import numpy as np
import pandas as pd
from experiment_core import TrainConfig,load_data,run_experiment,file_sha256,write_json
from research_runs import ROOT,PARTITIONS,source_hashes,canonical_hash,make_aggregator
from scenario_metrics import longer_scenarios,ClasswiseEvaluator
from research_analysis import cross_method_pair
from directional_study import checked_result as checked_artifacts
from strength_validation_study import STUDY as PARENT,hashes as parent_hashes,verify as verify_parent
from proximal_training import installed_trainer,TrainingTreatment
from proximal_analysis import analyse_study

STUDY=ROOT/'outputs/proximal_training/20261030_v1'
OLD=ROOT/'outputs/full_study/20261011_v1'
SCENARIOS=('clean','abrupt_shot_noise','gradual_stripe','label_flip','attack_stops')
SEEDS=(20261014,20261015)
CANDIDATES=('FedProx','FedProx+RFA')
MU=1.
NEW_SOURCES=('proximal_training.py','proximal_study.py','proximal_analysis.py')
TESTS=('test_proximal_training.py',)


def stamp():return datetime.now(timezone.utc).isoformat()
def hashes():return {**parent_hashes(),**{n:file_sha256(ROOT/'code'/n) for n in NEW_SOURCES}}
def base_key(j):return j['method'],j['regime'],j['scenario'],j['seed']


def jobs():
    common={'phase':'development','relative_rate':1.,'rounds':24,'train_images':512,'num_threads':4}
    result=[]
    for method,regime in (('FedAvg','IID'),('RFA','non-IID')):
        result.append({**common,'method':method,'regime':regime,'scenario':'clean','seed':SEEDS[0],
                       'mu':0.,'attack_policy':'compliant','role':'zero_mu_audit'})
    for method in CANDIDATES:
        for seed in SEEDS:
            for regime in PARTITIONS:
                for scenario in SCENARIOS:
                    policies=('compliant','bypass') if scenario in ('label_flip','attack_stops') else ('compliant',)
                    for policy in policies:
                        result.append({**common,'method':method,'regime':regime,'scenario':scenario,'seed':seed,
                                       'mu':MU,'attack_policy':policy,'role':'candidate'})
    assert len(result)==58
    return result


def baseline_records():
    found={}
    for path in (OLD/'jobs').glob('*/result.json'):
        r=json.loads(path.read_text());j=r['job']
        if (j['phase']!='development' or j['method'] not in ('FedAvg','RFA') or j['relative_rate']!=1.
                or j['seed'] not in SEEDS or j['scenario'] not in SCENARIOS
                or j.get('variant','full')!='full' or j.get('overrides')):continue
        if r['source_sha256']!=source_hashes():raise ValueError('Original baseline sources changed.')
        k=base_key(j)
        if k in found:raise ValueError('Duplicate baseline condition.')
        found[k]=checked_artifacts(r)
    expected={(m,r,s,seed) for m in ('FedAvg','RFA') for r in PARTITIONS for s in SCENARIOS for seed in SEEDS}
    if set(found)!=expected:raise ValueError('Missing required baseline comparisons.')
    return list(found.values())


def prepare(study):
    study.mkdir(parents=True,exist_ok=True)
    if (study/'protocol.json').exists():verify(study);return
    verify_parent(PARENT)
    baselines=baseline_records();write_json(study/'reused_manifest.json',{'records':baselines})
    p={'created_at':stamp(),'status':'declared_before_training',
       'authorization':'User: ok lets experiment; matched FedAvg/FedProx/RFA/FedProx+RFA with attackers allowed to ignore local penalty.',
       'scope':'Exploratory end-to-end development trajectories, two reused seeds and fixed partitions. Not a tuned FedProx benchmark, new FedReTrust version, confirmation or novelty claim.',
       'source_sha256':hashes(),'parent_protocol_sha256':file_sha256(PARENT/'protocol.json'),
       'reused_manifest_sha256':file_sha256(study/'reused_manifest.json'),
       'partition_sha256':{r:file_sha256(ROOT/'data/partitions'/part/'indices.json') for r,part in PARTITIONS.items()},
       'jobs':jobs(),'new_runs':58,'candidate_runs':56,'zero_mu_audit_runs':2,'reused_baseline_runs':40,
       'workload':{'rounds':24,'clients':10,'training_images_per_client':512,'local_epochs':1,
                   'batch_size':64,'learning_rate':.01,'momentum':.9,'evaluation_examples_per_task':1000,
                   'evaluation_tasks':['clean','shot_noise_s3','stripe'],'max_workers':2,'cpu_threads_per_worker':4},
       'local_objective':{'formula':'mean minibatch cross entropy + mu/2 * sum_all_trainable_parameters((local-current_broadcast)^2)',
                          'mu':MU,'parameter_count_normalization':False,'anchor':'detached current broadcast, reset every client/round',
                          'mu_choice':'One fixed coefficient from the author-recommended search range, chosen before new results. No claim that it is optimal. No post-result sweep in this pilot.',
                          'optimizer_distinction':'Use original local SGD with momentum0.9 to isolate the objective change. Author reference PGD uses a different optimizer setup; this is a FedProx-objective adaptation, not an exact paper benchmark reproduction.',
                          'training_loss_reporting':'online_train_loss remains supervised CE only; record proximal penalty and total objective separately.'},
       'attack_policies':{'compliant':'All clients, including those training on flipped labels, use the local proximal objective.',
                          'bypass':'Only clients with active label poisoning use mu=0. They resume the normal proximal objective after the attack ends. This is attacker simulation, not server access to actor labels.'},
       'aggregation':'Original FedAvg or original RFA (same solver and server rate1) without any new validation-report interface. Aggregation receives only submitted states, IDs, example counts and broadcast.',
       'data_access':'No trusted server reference images; the reserved300 remain excluded and unused. Standard development task evaluation is offline reporting only and never enters training or aggregation. No official-test loader.',
       'scenarios':{s:longer_scenarios()[s].to_dict() for s in SCENARIOS},
       'not_in_this_pilot':'Simultaneous drift+attack, Gaussian updates and recurring attacks are not new training conditions in this bounded five-scenario comparison.',
       'integrity':'Zero-mu FedAvg IID and RFA non-IID, seed20261014, must exactly reproduce all24 original global states, all task metrics and all submitted client states. Freeze sources/inputs and require matched cross-method config, data, initialization and evaluation provenance.',
       'metrics':'Original class/population-weighted legitimate accuracy: mean rounds5-24, affected and unchanged clients, tail21-24, final24, attack-active accuracy and own-method clean-relative damage. Raw task accuracy/loss, local CE/penalty, contributions and compute are retained.',
       'recovery':'After attacks stop at10, accuracy must be at least own paired clean minus1pp for3 consecutive rounds in11-24. Report streak start, confirmation and censoring for all clients and former attackers. No stateless readmission claim.',
       'target':'Descriptive two-seed means: each proximal treatment must match FedAvg on clean, honest noise/stripe affected accuracy and unchanged-client drift accuracy; match RFA on persistent attack absolute accuracy and clean-relative damage under BOTH attacker policies, in BOTH regimes. Strict zero shortfall; per-seed deltas and recovery reported separately. A small numerical difference is not a statistical superiority claim.',
       'stopping':'Exactly58 new trajectories; no post-hoc mu tuning, optimizer changes, best-round selection or official-test evaluation. Report negative results and every declared policy.',
       'sources':['https://arxiv.org/abs/1812.06127','https://github.com/litian96/FedProx','https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py']}
    write_json(study/'protocol.json',p)
    snapshot=study/'source_snapshot';snapshot.mkdir()
    for name in hashes():shutil.copy2(ROOT/'code'/name,snapshot/name)
    for name in TESTS:shutil.copy2(ROOT/'code/tests'/name,snapshot/name)
    print(json.dumps({'new_runs':58,'candidate_runs':56,'verified_reused_baselines':40,'mu':MU}),flush=True)


def verify(study):
    p=json.loads((study/'protocol.json').read_text())
    if p['source_sha256']!=hashes():raise ValueError('Proximal sources changed after freeze.')
    if file_sha256(PARENT/'protocol.json')!=p['parent_protocol_sha256']:raise ValueError('Parent protocol changed.')
    if file_sha256(study/'reused_manifest.json')!=p['reused_manifest_sha256']:raise ValueError('Baseline manifest changed.')
    for regime,digest in p['partition_sha256'].items():
        if file_sha256(ROOT/'data/partitions'/PARTITIONS[regime]/'indices.json')!=digest:raise ValueError('Partition changed.')
    verify_parent(PARENT)
    return p


def checked_result(r):
    checked_artifacts(r)
    if r['job'] not in jobs() or r['source_sha256']!=hashes():raise ValueError('Unregistered proximal run.')
    return r


def new_records(study):
    return [checked_result(json.loads(p.read_text())) for p in sorted((study/'jobs').glob('*/result.json'))]


def audit_zero(result,reference):
    checked_artifacts(reference);cross_method_pair(result,reference)
    path,old=ROOT/result['run_path'],ROOT/reference['run_path']
    a,b=pd.read_csv(path/'history.csv'),pd.read_csv(old/'history.csv')
    if a.global_state_sha256.tolist()!=b.global_state_sha256.tolist():raise AssertionError('Zero-mu global trajectory changed.')
    pd.testing.assert_frame_equal(pd.read_csv(path/'task_metrics.csv'),pd.read_csv(old/'task_metrics.csv'),check_exact=True)
    a,b=pd.read_csv(path/'client_metrics.csv'),pd.read_csv(old/'client_metrics.csv')
    for field in ('submitted_state_sha256','broadcast_state_sha256','shuffle_seed','optimizer_steps','online_train_loss'):
        if a[field].tolist()!=b[field].tolist():raise AssertionError('Zero-mu local path changed: '+field)
    return {'job_id':result['job_id'],'reference_job_id':reference['job_id'],'global_states_identical':24,
            'submitted_states_identical':240,'all_task_metrics_identical':True}


def worker(j,study):
    verify(study)
    if j not in jobs():raise ValueError('Undeclared job.')
    jid=canonical_hash({'job':j,'sources':hashes()})[:20]
    directory=study/'jobs'/jid;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'result.json').exists():return checked_result(json.loads((directory/'result.json').read_text()))
        if (directory/'failure.json').exists():raise ValueError('Investigate preserved failure before resuming.')
        write_json(directory/'request.json',{'job':j,'sources':hashes()});started=time.perf_counter()
        try:
            cfg=TrainConfig(partition_id=PARTITIONS[j['regime']],development_seed=j['seed'],rounds=24,
                            max_train_per_client=512,max_validation=1000,num_threads=4)
            data=load_data(ROOT,cfg)
            base_method='RFA' if j['method'] in ('RFA','FedProx+RFA') else 'FedAvg'
            aggregator=make_aggregator(base_method,data,cfg)
            if j['role']=='candidate':aggregator=TrainingTreatment(aggregator,j['method'],j['mu'])
            with installed_trainer(j['mu'],j['attack_policy']):
                run=run_experiment(data,cfg,longer_scenarios()[j['scenario']],aggregator,
                                   notebook_code_sha256=jid,verbose=False,evaluator=ClasswiseEvaluator())
            counts={i:np.bincount(data.pool.targets[ix].numpy(),minlength=10).tolist()
                    for i,ix in data.manifest['indices']['clients'].items()}
            write_json(run/'population_class_counts.json',counts)
            write_json(run/'training_protocol.json',{'job':j,'source_sha256':hashes(),
                       'injection':'Process-local temporary train_client replacement; frozen experiment_core file unchanged; original callable restored.',
                       'client_private_data_only':True,'trusted_server_images':0,'official_test_used':False,
                       'global_development_evaluation':'reporting_only_no_feedback'})
            names=('run.json','selection.json','summary.json','history.csv','task_metrics.csv','client_metrics.csv',
                   'aggregation_diagnostics.json','population_class_counts.json','training_protocol.json',
                   'initial_global_model_state.pt','final_global_model_state.pt','status.json','truth/scenario_truth.csv')
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result={'job_id':jid,'job':j,'status':'complete','run_path':str(run.relative_to(ROOT)),
                    'artifact_sha256':{n:file_sha256(run/n) for n in names},'source_sha256':hashes(),
                    'worker_wall_seconds':time.perf_counter()-started,
                    'process_lifetime_peak_rss_bytes':int(rss if platform.system()=='Darwin' else rss*1024)}
            if j['role']=='zero_mu_audit':
                refs=json.loads((study/'reused_manifest.json').read_text())['records']
                ref=next(r for r in refs if base_key(r['job'])==base_key(j))
                result['zero_mu_audit']=audit_zero(result,ref)
            write_json(directory/'result.json',result);return result
        except BaseException as exc:
            write_json(directory/'failure.json',{'job':j,'error':repr(exc),'at':stamp()});raise


def run(study):
    verify(study)
    with (study/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(study/'status.json',{'status':'running','at':stamp()})
        requests=study/'requests';requests.mkdir(exist_ok=True)
        def launch(j):
            ident=canonical_hash(j)[:20];path=requests/(ident+'.json');write_json(path,j)
            with (requests/(ident+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--study',str(study),'--job',str(path)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            return j
        try:
            for role in ('zero_mu_audit','candidate'):
                have={canonical_hash(r['job']) for r in new_records(study)}
                pending=[j for j in jobs() if j['role']==role and canonical_hash(j) not in have]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for future in as_completed([pool.submit(launch,j) for j in pending]):
                        done=future.result();completed=len(new_records(study))
                        progress={'completed':completed,'total':58,'stage':role,'job':done,'at':stamp()}
                        write_json(study/'progress.json',progress);print(json.dumps(progress),flush=True)
            records=new_records(study)
            if len(records)!=58:raise ValueError('Incomplete training study.')
            baselines=json.loads((study/'reused_manifest.json').read_text())['records']
            write_json(study/'result_manifest.json',{'records':[r for r in records if r['job']['role']=='candidate']+baselines,
                'new_records':records,'zero_mu_records':[r for r in records if r['job']['role']=='zero_mu_audit'],
                'reused_baseline_records':baselines,'at':stamp(),'official_test_used':False})
            analyse_study(study)
            write_json(study/'status.json',{'status':'complete','new_runs':58,'candidate_runs':56,'reused_baselines':40,'at':stamp()})
        except BaseException as exc:
            write_json(study/'status.json',{'status':'stopped_on_error','error':repr(exc),'at':stamp()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','worker','analyse'))
    p.add_argument('--study',type=Path,default=STUDY);p.add_argument('--job',type=Path);a=p.parse_args()
    if a.action=='worker':worker(json.loads(a.job.read_text()),a.study)
    elif a.action=='analyse':verify(a.study);analyse_study(a.study)
    else:globals()[a.action](a.study)
