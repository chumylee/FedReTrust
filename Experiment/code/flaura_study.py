"""Frozen FLAURA reconstruction comparison, separate from all prior studies."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse, fcntl, json, platform, resource, shutil, subprocess, sys, time
import numpy as np
from experiment_core import TrainConfig, load_data, run_experiment, file_sha256, write_json
from research_runs import ROOT, PARTITIONS, canonical_hash
from scenario_metrics import ContributionRecorder, ClasswiseEvaluator, longer_scenarios
from directional_study import checked_result as checked_artifacts
from blockwise_study import STUDY as PARENT, hashes as parent_hashes, verify as verify_parent
from flaura import FLAURAConfig, FLAURAAggregator, PLRCalibrationObserver, build_auxiliary_reference

STUDY = ROOT/'outputs/flaura/20261030_v1'
MAIN = 'FLAURA-reconstruction'
CONTROL = 'FLAURA-no-filter-control'
CALIBRATION = 'FedAvg-PLR-calibration'
BASELINES = ('FedAvg','RFA','FLTrust','Viceroy','FedReTrust','FedReTrust-RF-recovery','FedReTrust-Direction-loose','FedReTrust-v04')
SEEDS = (20261014,20261015)
CALIBRATION_SEED = 20261013
CORE = ('clean','gradual_stripe','label_flip')
REPEAT = ('clean','abrupt_shot_noise','gradual_stripe','label_flip','attack_stops')
SCENARIOS = tuple(s for s in longer_scenarios() if s!='ambiguous_malicious_input')
SETTINGS = {
    MAIN: {'sigma_factor':1.,'gamma':.8}, CONTROL:{'sigma_factor':1.,'gamma':0.},
    'FLAURA-sigma-half':{'sigma_factor':.5,'gamma':.8},
    'FLAURA-sigma-double':{'sigma_factor':2.,'gamma':.8},
    'FLAURA-gamma-0.5':{'sigma_factor':1.,'gamma':.5},
    'FLAURA-gamma-1.0':{'sigma_factor':1.,'gamma':1.},
}
NEW_SOURCES = ('flaura.py','flaura_study.py','flaura_analysis.py','research_analysis.py','research_study.py',
               'tests/test_flaura.py','tests/test_flaura_study.py')

def stamp(): return datetime.now(timezone.utc).isoformat()
def hashes(): return {**parent_hashes(),**{n:file_sha256(ROOT/'code'/n) for n in NEW_SOURCES}}
def key(j): return j['method'],j['regime'],j['scenario'],j['seed']
def job(method,regime,scenario,seed):
    return {'method':method,'regime':regime,'scenario':scenario,'seed':seed,'phase':'development',
            'relative_rate':1.,'root_steps':8,'rounds':6 if method==CALIBRATION else 24,
            'train_images':512,'num_threads':4,
            'flaura_stage':'calibration' if method==CALIBRATION else 'main' if method==MAIN else 'control' if method==CONTROL else 'sensitivity'}

def jobs():
    out=[job(CALIBRATION,g,'clean',CALIBRATION_SEED) for g in PARTITIONS]
    out += [job(MAIN,g,s,seed) for seed in SEEDS for g in PARTITIONS for s in (SCENARIOS if seed==SEEDS[0] else REPEAT)]
    out += [job(CONTROL,g,s,seed) for seed in SEEDS for g in PARTITIONS for s in CORE]
    out += [job(m,g,s,SEEDS[0]) for m in SETTINGS if m not in (MAIN,CONTROL) for g in PARTITIONS for s in CORE]
    assert len(out)==68
    return out

def prior_records():
    verify_parent(PARENT)
    m=json.loads((PARENT/'result_manifest.json').read_text())
    primary=[r for r in m['pilot_records'] if r['job']['method'] in BASELINES]
    context=m['context_records']
    expected={key(job(b,g,s,seed)) for b in BASELINES for seed in SEEDS for g in PARTITIONS for s in (SCENARIOS if seed==SEEDS[0] else REPEAT)}
    if {key(r['job']) for r in primary}!=expected or len(primary)!=240:
        raise ValueError('Incomplete prior comparison panel.')
    for r in primary+context:
        checked_artifacts(r)
        for name,h in r['source_sha256'].items():
            if file_sha256(ROOT/'code'/name)!=h:raise ValueError('Prior source changed: '+name)
    return primary,context

def prepare(study):
    study.mkdir(parents=True,exist_ok=True)
    if (study/'protocol.json').exists():verify(study);return
    prior,context=prior_records()
    write_json(study/'reused_manifest.json',{'records':prior,'context_records':context})
    resource_dir=ROOT/'resources/flaura_reference'
    source_files={str(p.relative_to(ROOT)):file_sha256(p) for p in resource_dir.iterdir() if p.is_file()}
    p={
        'created_at':stamp(),'status':'declared_before_calibration_and_training',
        'authorization':'Implement FLAURA and compare with the other baselines, one method at a time.',
        'source_sha256':hashes(),'parent_protocol_sha256':file_sha256(PARENT/'protocol.json'),
        'resource_sha256':source_files,'reused_manifest_sha256':file_sha256(study/'reused_manifest.json'),
        'partition_sha256':{g:file_sha256(ROOT/'data/partitions'/part/'indices.json') for g,part in PARTITIONS.items()},
        'jobs':jobs(),'new_runs':68,'calibration_runs':2,'new_24_round_comparisons':66,
        'main':MAIN,'control':CONTROL,'baselines':BASELINES,'settings':SETTINGS,
        'base_config':asdict(FLAURAConfig()),'reused_primary_runs':240,'reused_context_runs':len(context),
        'calibration':{'seed':CALIBRATION_SEED,'rounds':6,'regimes':list(PARTITIONS),
            'rule':'Fixed sigma_g equals median of 120 client-to-geometric-center Frobenius distances from two six-round clean FedAvg runs; floor 1e-12. Both regimes pooled. No accuracy or attack outcome used.',
            'purpose':'Numerical PLR scale calibration for an unspecified paper hyperparameter; an explicit offline clean-development privilege, not assumed knowledge of clean rounds in evaluated runs.'},
        'selection':'No outcome-based setting selection. Main remains alpha=.5 beta=.5 gamma=.8 and calibrated fixed sigma_g. Every sensitivity result retained; no best-setting substitution.',
        'method_status':'Documented Algorithm 1 reconstruction; no author repository or parameter defaults were located in the article or targeted public search.',
        'paper_url':'https://www.nature.com/articles/s41598-026-50985-2',
        'paper_vs_algorithm':{
            'global_score':'Use Algorithm 1 denominator 2*sigma_g^2; Eq.7 omits 2.',
            'threshold':'Use Algorithm 1 gamma*mean(updated trust); prose presents a global tau without its computation.',
            'feature_location':'Server recomputes PLRs from uploaded model weights and copied auxiliary images, following prose; Algorithm 1 instead shows clients uploading PLRs. No untrusted claimed PLRs are accepted.',
            'rebalancing':'Prose mentions variance-triggered rebalancing but gives no formula/threshold and Algorithm 1 omits it. Not invented or implemented.',
            'memory':'Use optional EMA initialized to 1. Fixed-rate EMA is symmetric, not inherently fast-penalty/slow-recovery.'},
        'unspecified_choices':{
            'alpha':.5,'beta':.5,'gamma':.8,'kernel':'RBF with pooled median positive squared-distance bandwidth, recomputed from current submitted PLRs',
            'MMD':'Biased empirical RKHS mean-embedding norm, square root of squared MMD, includes diagonal kernel terms',
            'knee':'Largest adjacent jump of sorted other-client MMD distances (Eq.11); equal-gap or flat profiles use all peers; cutoff ties included.',
            'median_solver':'Uniform-client smoothed Weiszfeld, 200 iterations maximum, smoothing 1e-8, relative objective tolerance 1e-9; convergence diagnostics retained.',
            'weighting':'Normalized trust only, not multiplied by sample count; zero retained trust freezes broadcast.'},
        'auxiliary_data':{'images':100,'per_class':10,'selection':'First ten sorted reserved server indices per class; fixed across scenarios/seeds; disjoint from all clients and development evaluation.',
            'labels':'Only used to select class-balanced images. Aggregator receives images and metadata, no labels.',
            'backpropagation':False,'reference_forward_examples_per_round':1000},
        'common_training':'Existing MNIST CNN; SGD lr .01, momentum .9; 10 clients, 512 images/client, one local epoch, batch64, 24 rounds. Same saved partitions and matched seed selections. This adapts the aggregation method to our common workload, not the paper Adam/five-epoch/FM NIST setup.',
        'scope':'Main all ten scenarios in seed20261014; five scenarios repeated in seed20261015. No-filter control clean/stripe/label in both seeds. Four one-at-a-time scale/threshold sensitivities on three core scenarios in first seed only.',
        'scenarios':{s:longer_scenarios()[s].to_dict() for s in SCENARIOS},
        'metrics':'Mean rounds5-24, final24, tail21-24; clean/noise/stripe; attack absolute accuracy and own-control signed damage; unchanged-client accuracy; online training losses; trust, exclusions, applied coefficients and contribution norms; model and influence restoration; overhead.',
        'recovery':'Within own matched control minus 1pp for three consecutive rounds after attack, retaining onset, confirmation, already-within-criterion and censoring.',
        'interpretation':'Lower drift accuracy alone does not establish harmful honest exclusion. Examine affected-client weights, exclusions and control differences, with both seeds. No-filter contrast isolates hard filtering along different trajectories; it does not establish useful updates from weight alone.',
        'statistics':'Two development seeds for principal outcomes; one for full scenario and sensitivity panels. Descriptive paired differences only; no significance claim and no client-round pseudoreplication.',
        'stopping':'Complete all 68 runs regardless of outcome. No FedCognis, new FedReTrust, post-result tuning or official-test access.',
    }
    write_json(study/'protocol.json',p)
    for name in hashes():
        dest=study/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'code'/name,dest)
    print(json.dumps({'prepared':str(study),'new_runs':68,'reused_primary':240,'context':len(context)}),flush=True)

def verify(study):
    p=json.loads((study/'protocol.json').read_text())
    if p['source_sha256']!=hashes():raise ValueError('FLAURA sources changed after freeze.')
    if p['parent_protocol_sha256']!=file_sha256(PARENT/'protocol.json'):raise ValueError('Parent protocol changed.')
    if p['reused_manifest_sha256']!=file_sha256(study/'reused_manifest.json'):raise ValueError('Reuse manifest changed.')
    for name,h in p['resource_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('Paper/specification artifact changed.')
    for g,h in p['partition_sha256'].items():
        if file_sha256(ROOT/'data/partitions'/PARTITIONS[g]/'indices.json')!=h:raise ValueError('Partition changed.')
    verify_parent(PARENT)
    return p

def checked_result(r):
    checked_artifacts(r)
    if r['source_sha256']!=hashes() or r['job'] not in jobs():raise ValueError('Unregistered FLAURA job.')
    return r

def records(study):
    return [checked_result(json.loads(p.read_text())) for p in sorted((study/'jobs').glob('*/result.json'))]

def freeze_scale(study):
    runs=[r for r in records(study) if r['job']['method']==CALIBRATION]
    if len(runs)!=2:raise ValueError('Both calibration regimes required.')
    values=[]
    for r in runs:
        for row in json.loads((ROOT/r['run_path']/'aggregation_diagnostics.json').read_text()):
            values.extend(row['values']['plr_center_distances'])
    if len(values)!=120 or not np.isfinite(values).all():raise ValueError('Calibration geometry incomplete.')
    report={'status':'frozen_from_clean_geometry_before_comparison','sigma_g':max(float(np.median(values)),1e-12),
            'rule':json.loads((study/'protocol.json').read_text())['calibration']['rule'],
            'jobs':[r['job_id'] for r in runs],'geometry':values,
            'artifact_sha256':{r['job_id']:r['artifact_sha256'] for r in runs},'protocol_sha256':file_sha256(study/'protocol.json')}
    path=study/'scale_freeze.json'
    if path.exists():
        if json.loads(path.read_text())!=report:raise ValueError('Calibration freeze changed.')
    else:write_json(path,report)
    return report

def worker(j,study):
    verify(study)
    if j not in jobs():raise ValueError('Undeclared job.')
    scale=None if j['method']==CALIBRATION else freeze_scale(study)
    jid=canonical_hash({'job':j,'sources':hashes(),'scale':scale})[:20]
    directory=study/'jobs'/jid;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'result.json').exists():return checked_result(json.loads((directory/'result.json').read_text()))
        if (directory/'failure.json').exists():raise ValueError('Investigate preserved failure before retrying.')
        write_json(directory/'request.json',{'job':j,'source_sha256':hashes()})
        started=time.perf_counter()
        try:
            cfg=TrainConfig(partition_id=PARTITIONS[j['regime']],development_seed=j['seed'],rounds=j['rounds'],max_train_per_client=512,max_validation=1000,num_threads=4)
            data=load_data(ROOT,cfg);reference=build_auxiliary_reference(data)
            if j['method']==CALIBRATION:base=PLRCalibrationObserver(reference)
            else:
                setting=SETTINGS[j['method']]
                base=FLAURAAggregator(reference,FLAURAConfig(sigma_g=scale['sigma_g']*setting['sigma_factor'],gamma=setting['gamma']),name=j['method'])
            run=run_experiment(data,cfg,longer_scenarios()[j['scenario']],ContributionRecorder(base),notebook_code_sha256=jid,verbose=False,evaluator=ClasswiseEvaluator())
            counts={i:np.bincount(data.pool.targets[ix].numpy(),minlength=10).tolist() for i,ix in data.manifest['indices']['clients'].items()}
            write_json(run/'population_class_counts.json',counts)
            write_json(run/'flaura_access.json',{'job':j,'reference':reference.metadata,'official_test_used':False,
                'client_local_training_unchanged':True,'scale_freeze_sha256':file_sha256(study/'scale_freeze.json') if scale else None,
                'auxiliary_labels_in_aggregation':False,'server_backpropagation':False})
            names=('run.json','selection.json','summary.json','history.csv','task_metrics.csv','client_metrics.csv','aggregation_diagnostics.json',
                   'population_class_counts.json','flaura_access.json','initial_global_model_state.pt','final_global_model_state.pt','status.json','truth/scenario_truth.csv')
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result={'job_id':jid,'job':j,'status':'complete','run_path':str(run.relative_to(ROOT)),'source_sha256':hashes(),
                'artifact_sha256':{n:file_sha256(run/n) for n in names},'worker_wall_seconds':time.perf_counter()-started,
                'process_lifetime_peak_rss_bytes':int(rss if platform.system()=='Darwin' else rss*1024)}
            write_json(directory/'result.json',result)
            return result
        except BaseException as exc:
            write_json(directory/'failure.json',{'job':j,'error':repr(exc),'at':stamp()});raise

def run(study):
    from flaura_analysis import analyse
    verify(study)
    with (study/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(study/'status.json',{'status':'running','at':stamp()})
        requests=study/'requests';requests.mkdir(exist_ok=True)
        def launch(j):
            ident=canonical_hash(j)[:20];path=requests/(ident+'.json');write_json(path,j)
            with (requests/(ident+'.log')).open('w') as out:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--study',str(study),'--job',str(path)],stdout=out,stderr=subprocess.STDOUT,check=True)
            return j
        def stage(label):
            expected=[j for j in jobs() if (j['method']==CALIBRATION)==(label=='calibration')]
            have={key(r['job']) for r in records(study)}
            with ThreadPoolExecutor(max_workers=2) as pool:
                for f in as_completed([pool.submit(launch,j) for j in expected if key(j) not in have]):
                    j=f.result();have.add(key(j))
                    progress={'stage':label,'completed':sum(key(x) in have for x in expected),'total':len(expected),'job':j,'at':stamp()}
                    write_json(study/'progress.json',progress);print(json.dumps(progress),flush=True)
        try:
            stage('calibration');scale=freeze_scale(study);print(json.dumps({'fixed_sigma_g':scale['sigma_g']}),flush=True)
            stage('comparison');new=records(study);prior=json.loads((study/'reused_manifest.json').read_text())
            scientific=[r for r in new if r['job']['method']!=CALIBRATION]
            write_json(study/'result_manifest.json',{'records':scientific+prior['records'],'new_records':new,'context_records':prior['context_records'],
                'calibration_records':[r for r in new if r['job']['method']==CALIBRATION],'at':stamp()})
            analyse(study)
            write_json(study/'status.json',{'status':'complete','new_runs':68,'at':stamp()})
        except BaseException as exc:
            write_json(study/'status.json',{'status':'stopped_on_error','error':repr(exc),'at':stamp()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','worker','analyse'))
    p.add_argument('--study',type=Path,default=STUDY);p.add_argument('--job',type=Path);args=p.parse_args()
    if args.action=='worker':worker(json.loads(args.job.read_text()),args.study)
    elif args.action=='analyse':
        from flaura_analysis import analyse
        verify(args.study);analyse(args.study)
    else:globals()[args.action](args.study)
