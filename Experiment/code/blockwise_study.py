"""Frozen v0.4 pilot with a declared fresh-seed continuation, no earlier edits."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,fcntl,json,platform,resource,shutil,subprocess,sys,time
import numpy as np
from experiment_core import TrainConfig,load_data,run_experiment,file_sha256,write_json
from research_runs import ROOT,PARTITIONS,canonical_hash,make_aggregator,source_hashes
from scenario_metrics import ClasswiseEvaluator,ContributionRecorder,longer_scenarios
from directional_study import checked_result as checked_artifacts
from fedretrust_reference_free import ReferenceFreeFedReTrust
from fedretrust_directional import DirectionalFedReTrust
from fedretrust_blockwise import BlockwiseConfig,BlockwiseFedReTrust
from blockwise_recording import BlockContributionRecorder
from proximal_study import STUDY as PARENT,hashes as parent_hashes,verify as verify_parent

STUDY=ROOT/'outputs/reference_free/20261030_v04'
MAIN='FedReTrust-v04';CONTROL='FedReTrust-v04-block-RFA-control'
BASELINES=('FedAvg','RFA','FLTrust','Viceroy','FedReTrust','FedReTrust-RF-recovery','FedReTrust-Direction-loose')
CORE=('clean','gradual_stripe','label_flip')
REPEAT=('clean','abrupt_shot_noise','gradual_stripe','label_flip','attack_stops')
SCENARIOS=tuple(s for s in longer_scenarios() if s!='ambiguous_malicious_input')
SEEDS=(20261014,20261015)
FRESH_SEEDS=tuple(range(20262001,20262006))
NEW_SOURCES=('fedretrust_blockwise.py','blockwise_recording.py','blockwise_study.py','blockwise_analysis.py')

def stamp():return datetime.now(timezone.utc).isoformat()
def hashes():return {**parent_hashes(),**{n:file_sha256(ROOT/'code'/n) for n in NEW_SOURCES}}
def key(j):return j['method'],j['regime'],j['scenario'],j['seed']
def job(method,regime,scenario,seed,stage='pilot'):
    return {'method':method,'regime':regime,'scenario':scenario,'seed':seed,'phase':'development',
            'relative_rate':1.,'root_steps':8,'rounds':24,'train_images':512,'num_threads':4,'v04_stage':stage}

def pilot_jobs():
    out=[job(MAIN,g,s,seed) for seed in SEEDS for g in PARTITIONS for s in (SCENARIOS if seed==SEEDS[0] else REPEAT)]
    out += [job(CONTROL,g,s,seed) for seed in SEEDS for g in PARTITIONS for s in CORE]
    out += [job(m,g,s,SEEDS[1]) for m in BASELINES[-2:] for g in PARTITIONS for s in ('abrupt_shot_noise','attack_stops')]
    assert len(out)==50;return out

def validation_jobs():
    return [job(m,g,s,seed,'fresh_seed_validation') for seed in FRESH_SEEDS for m in (MAIN,)+BASELINES for g in PARTITIONS for s in CORE]

def expected_pilot_keys():
    return {key(job(m,g,s,seed)) for m in (MAIN,)+BASELINES for seed in SEEDS for g in PARTITIONS
            for s in (SCENARIOS if seed==SEEDS[0] else REPEAT)} | {key(j) for j in pilot_jobs() if j['method']==CONTROL}

def prior_records():
    found={}
    locations=('full_study/20261011_v1','reference_free/20261030_v02','reference_free/20261030_v03')
    needed=expected_pilot_keys()-{key(j) for j in pilot_jobs()}
    for location in locations:
        for p in (ROOT/'outputs'/location/'jobs').glob('*/result.json'):
            r=json.loads(p.read_text());j=r['job']
            if (j['phase']!='development' or j['relative_rate']!=1. or j.get('variant','full')!='full'
                    or j.get('overrides') or key(j) not in needed):continue
            checked_artifacts(r)
            if any(file_sha256(ROOT/'code'/name)!=digest for name,digest in r['source_sha256'].items()):raise ValueError('Earlier source changed.')
            if key(j) in found and found[key(j)]['job_id']!=r['job_id']:raise ValueError('Conflicting prior trajectory.')
            found[key(j)]=r
    if set(found)!=needed:raise ValueError('Missing prior comparisons: '+str(needed-set(found)))
    return list(found.values())

def context_records():
    p=json.loads((PARENT/'result_manifest.json').read_text())
    out=[r for r in p['records'] if r['job']['method'] in ('FedProx','FedProx+RFA')]
    m=json.loads((ROOT/'outputs/mechanism_pilot/20261030_v1/result_manifest.json').read_text())
    out += [r for r in m['observed_records'] if r['job']['method']!='FedAvg']
    for r in out:checked_artifacts(r)
    return out

def prepare(study):
    study.mkdir(parents=True,exist_ok=True)
    if (study/'protocol.json').exists():verify(study);return
    verify_parent(PARENT);prior=prior_records();context=context_records()
    write_json(study/'reused_manifest.json',{'records':prior,'context_records':context})
    p={'created_at':stamp(),'status':'declared_before_training','authorization':'User requested version0.4 and comparison with the rest.',
        'source_sha256':hashes(),'parent_protocol_sha256':file_sha256(PARENT/'protocol.json'),
        'reused_manifest_sha256':file_sha256(study/'reused_manifest.json'),
        'partition_sha256':{g:file_sha256(ROOT/'data/partitions'/part/'indices.json') for g,part in PARTITIONS.items()},
        'main':MAIN,'control':CONTROL,'config':asdict(BlockwiseConfig()),'baselines':BASELINES,
        'pilot_jobs':pilot_jobs(),'pilot_new_runs':50,'pilot_reused_runs':len(prior),'pilot_scientific_trajectories':252,
        'context_only_trajectories':len(context),
        'workload':'24 rounds;10 clients;512 images/client;1 epoch;batch64;SGD lr0.01 momentum0.9;1000 development images/task;2 workers with4 CPU threads each.',
        'mechanism':'Split every weight+bias module into a block. Norm-cap each update at6 times the block median norm. Compute original3-step RFA separately per block. Classifier uses this RFA result. Feature blocks add sample-weighted residuals clipped at6 times median residual distance. Residual allowance falls immediately and rises by at most0.5 per supportive participation. No directional intent classifier and no assumed clean warm-up.',
        'formula':'bounded_i=c_i*d_i;center=sum(r_i*bounded_i);q_i=min(1,radius/distance_i,previous_q_i+0.5);g=center+sum(a_i*q_i*(bounded_i-center)). Classifier/control q=0. Effective RAW block coefficient=c_i*(a_i*q_i+(1-sum(a*q))*r_i).',
        'parameter_choice':'One fixed permissive feature radius6, chosen before results to test retention of minority feature adaptation while the classifier remains robust. Norm cap6 and restoration0.5 reuse earlier constants. No post-result tuning or swapping in the control.',
        'control_definition':'Block-RFA control shares the norm caps, partition, head and solver; sets all feature residual corrections to zero. It is a component control, not a second selected v0.4 candidate.',
        'coefficient_audit':'Per-block effective raw-update coefficients and actual global contribution norms/Gram matrix. No global scalar client coefficient exists; original client_metrics CSV coefficient field remains blank, never zero or an invented mean.',
        'data_access':'v0.4 and its control receive only submitted states/counts, IDs, broadcast, round and fixed scalar settings. Zero trusted server images or client validation reports. Original FLTrust/v0.1 comparator privilege remains explicit. Development evaluation is reporting only. No official-test loader.',
        'scenarios':{s:longer_scenarios()[s].to_dict() for s in SCENARIOS},
        'pilot_scope':'All10 original scenarios on seed20261014;5 clean/noise/stripe/label/stopping scenarios on seed20261015;control clean/stripe/label on both. Complete the two older reference-free versions on the second-seed noise/stopping cells.',
        'primary_metrics':'Mean rounds5-24 clean and honest-stripe affected accuracy; persistent attack signed clean-relative damage and absolute legitimate accuracy. Also report noise, unchanged clients, tail21-24, final24, all scenarios, losses, contributions, recovery and cost.',
        'recovery':'Own matched clean/no-attack scenario; within1pp for3 consecutive post-event rounds, preserving streak start, confirmation, attack-end status and censoring. No stateless readmission claim.',
        'continuation_gate':'Both regimes: v0.4 clean accuracy >= best of7 earlier comparators; stripe affected accuracy >= best of7; label damage <= best of7; absolute label-attack accuracy >= best of7. All8 strict two-seed mean checks must pass, plus >=1pp stripe gain over RFA in each regime. This gate selects continuation, not statistical superiority. The control cannot replace the main candidate.',
        'conditional_validation':{'trigger':'Only when the pilot continuation gate passes. Otherwise retain full negative pilot and stop.',
            'jobs':validation_jobs(),'runs':240,'seeds':FRESH_SEEDS,
            'scope':'Fixed v0.4 and all7 earlier methods;clean/stripe/label;both regimes;5 new paired seeds. Same reused dataset/partitions, so seed replication is not independent dataset confirmation.',
            'statistics':'Per-seed paired differences;Student-t95% intervals(df4). One-sided paired t tests for clean/stripe/absolute attacked accuracy versus v0.1/v0.2/v0.3 in both regimes;Holm correction across18 contrasts. Statistical improvement requires adjusted p<0.05;practical improvement additionally requires mean gain>=1pp. Remaining baseline comparisons descriptive. No client-round pseudoreplication.'},
        'stopping':'Exactly50 new pilot runs;only declared240-run continuation if gate passes. No adaptive variants, threshold sweeps, official-test access or claims that a failed setting proves impossibility.',
        'sources':['https://proceedings.mlr.press/v139/karimireddy21a.html','https://arxiv.org/abs/2212.01976','https://arxiv.org/abs/2207.00872','https://arxiv.org/abs/2207.01982'],
        'attribution':'Inspired by block-dependent aggregation, output-layer attack observations and robust clipping. This specific protected-head residual rule is not a reproduction of those algorithms and inherits no theorem. Previous scalar layer-detection failure is not re-labelled as positive evidence; this tests a different aggregation action.'}
    write_json(study/'protocol.json',p);snapshot=study/'source_snapshot';snapshot.mkdir()
    for n in hashes():shutil.copy2(ROOT/'code'/n,snapshot/n)
    for n in ('test_fedretrust_blockwise.py','test_blockwise_study.py'):shutil.copy2(ROOT/'code/tests'/n,snapshot/n)
    print(json.dumps({'new_pilot_runs':50,'reused_comparisons':len(prior),'context_comparisons':len(context)}),flush=True)

def verify(study):
    p=json.loads((study/'protocol.json').read_text())
    if p['source_sha256']!=hashes():raise ValueError('v0.4 sources changed after freeze.')
    if file_sha256(PARENT/'protocol.json')!=p['parent_protocol_sha256']:raise ValueError('Parent protocol changed.')
    if file_sha256(study/'reused_manifest.json')!=p['reused_manifest_sha256']:raise ValueError('Prior manifest changed.')
    for g,digest in p['partition_sha256'].items():
        if file_sha256(ROOT/'data/partitions'/PARTITIONS[g]/'indices.json')!=digest:raise ValueError('Partition changed.')
    verify_parent(PARENT);return p

def checked_result(r):
    checked_artifacts(r)
    if r['source_sha256']!=hashes() or r['job'] not in pilot_jobs()+validation_jobs():raise ValueError('Unregistered v0.4 study run.')
    return r

def records(study):return [checked_result(json.loads(p.read_text())) for p in sorted((study/'jobs').glob('*/result.json'))]

def worker(j,study):
    verify(study)
    if j not in pilot_jobs()+validation_jobs():raise ValueError('Undeclared job.')
    if j['v04_stage']=='fresh_seed_validation':
        if not json.loads((study/'analysis/pilot/summary.json').read_text())['continuation']['pass']:raise ValueError('Continuation gate failed.')
    jid=canonical_hash({'job':j,'sources':hashes()})[:20];directory=study/'jobs'/jid;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'result.json').exists():return checked_result(json.loads((directory/'result.json').read_text()))
        if (directory/'failure.json').exists():raise ValueError('Investigate preserved failure before retrying.')
        write_json(directory/'request.json',{'job':j,'source_sha256':hashes()});start=time.perf_counter()
        try:
            cfg=TrainConfig(partition_id=PARTITIONS[j['regime']],development_seed=j['seed'],rounds=24,max_train_per_client=512,max_validation=1000,num_threads=4)
            data=load_data(ROOT,cfg);method=j['method']
            if method in (MAIN,CONTROL):agg=BlockContributionRecorder(BlockwiseFedReTrust(BlockwiseConfig(all_block_rfa=method==CONTROL)))
            elif method=='FedReTrust-RF-recovery':agg=ContributionRecorder(ReferenceFreeFedReTrust())
            elif method=='FedReTrust-Direction-loose':agg=ContributionRecorder(DirectionalFedReTrust())
            else:agg=make_aggregator(method,data,cfg)
            run=run_experiment(data,cfg,longer_scenarios()[j['scenario']],agg,notebook_code_sha256=jid,verbose=False,evaluator=ClasswiseEvaluator())
            counts={i:np.bincount(data.pool.targets[ix].numpy(),minlength=10).tolist() for i,ix in data.manifest['indices']['clients'].items()}
            write_json(run/'population_class_counts.json',counts)
            write_json(run/'v04_training_protocol.json',{'job':j,'source_sha256':hashes(),'original_local_training_unchanged':True,
                'trusted_server_images':300 if method in ('FLTrust','FedReTrust') else 0,'official_test_used':False})
            names=('run.json','selection.json','summary.json','history.csv','task_metrics.csv','client_metrics.csv','aggregation_diagnostics.json',
                   'population_class_counts.json','v04_training_protocol.json','initial_global_model_state.pt','final_global_model_state.pt','status.json','truth/scenario_truth.csv')
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            r={'job_id':jid,'job':j,'status':'complete','run_path':str(run.relative_to(ROOT)),'source_sha256':hashes(),
                'artifact_sha256':{n:file_sha256(run/n) for n in names},'worker_wall_seconds':time.perf_counter()-start,
                'process_lifetime_peak_rss_bytes':int(rss if platform.system()=='Darwin' else rss*1024)}
            write_json(directory/'result.json',r);return r
        except BaseException as exc:write_json(directory/'failure.json',{'job':j,'error':repr(exc),'at':stamp()});raise

def run(study):
    from blockwise_analysis import analyse_study
    verify(study)
    with (study/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(study/'status.json',{'status':'running','at':stamp()});requests=study/'requests';requests.mkdir(exist_ok=True)
        def launch(j):
            ident=canonical_hash(j)[:20];p=requests/(ident+'.json');write_json(p,j)
            with (requests/(ident+'.log')).open('w') as out:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--study',str(study),'--job',str(p)],stdout=out,stderr=subprocess.STDOUT,check=True)
            return j
        def stage(jobs,label):
            have={key(r['job']) for r in records(study)};pending=[j for j in jobs if key(j) not in have]
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in as_completed([pool.submit(launch,j) for j in pending]):
                    j=future.result();n=sum(r['job']['v04_stage']==label for r in records(study))
                    progress={'stage':label,'completed':n,'total':len(jobs),'job':j,'at':stamp()}
                    write_json(study/'progress.json',progress);print(json.dumps(progress),flush=True)
        try:
            stage(pilot_jobs(),'pilot');new=records(study);prior=json.loads((study/'reused_manifest.json').read_text())
            pilot=[r for r in new if r['job']['v04_stage']=='pilot']
            write_json(study/'result_manifest.json',{'pilot_records':pilot+prior['records'],'new_records':new,'context_records':prior['context_records'],
                'validation_records':[r for r in new if r['job']['v04_stage']=='fresh_seed_validation'],'at':stamp()})
            summary=analyse_study(study,'pilot')
            if summary['continuation']['pass']:
                stage(validation_jobs(),'fresh_seed_validation');new=records(study)
                write_json(study/'result_manifest.json',{'pilot_records':pilot+prior['records'],'new_records':new,'context_records':prior['context_records'],
                    'validation_records':[r for r in new if r['job']['v04_stage']=='fresh_seed_validation'],'at':stamp()})
                analyse_study(study,'fresh_seed_validation')
            write_json(study/'status.json',{'status':'complete','pilot_new_runs':50,'validation_new_runs':240 if summary['continuation']['pass'] else 0,'at':stamp()})
        except BaseException as exc:write_json(study/'status.json',{'status':'stopped_on_error','error':repr(exc),'at':stamp()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','worker','analyse'))
    p.add_argument('--study',type=Path,default=STUDY);p.add_argument('--job',type=Path);p.add_argument('--stage',default='pilot');a=p.parse_args()
    if a.action=='worker':worker(json.loads(a.job.read_text()),a.study)
    elif a.action=='analyse':
        from blockwise_analysis import analyse_study
        verify(a.study);analyse_study(a.study,a.stage)
    else:globals()[a.action](a.study)
