"""Artifact-only presentation of the completed client-local validation pilot.

This builder is outside the frozen experimental source set. It never trains,
evaluates model checkpoints, chooses scoring rules or changes a declared gate.
"""
from pathlib import Path
import argparse
import json
import os

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "outputs/client_validation/20261030_v1"
DEFAULT_NOTEBOOK = ROOT / "code/21_client_local_validation.ipynb"


INTRODUCTION = """# 21 — Client-local validation of proposed updates

Can peer validation supply useful evidence that update geometry missed? This pilot measures three fixed scoring rules on client-local held-out data, including dishonest validation reports. It does not train a global model with a new aggregation rule. Its outputs are validation decisions and independently audited loss changes, not end-to-end accuracy improvements.

The 96 cases combine two fixed partitions, two previously used development seeds, four common clean broadcasts and six scenarios. The candidate step is deliberately small: `broadcast + 0.1 × local update`. Each candidate is evaluated by ten clients, with its own report excluded from its peer score. No server-owned reference images or global development-evaluator feedback enter the scoring rule.

This is an exploratory feasibility study. It is not a FLShield reproduction, a production privacy guarantee, an independent confirmation study or a novelty claim."""

METHOD = r"""## Fixed scoring rules, held-out roles and dishonest reports

The primary utility observation is the improvement in normalized Brier loss,

$$d_{iv}=\mathbb E_{V_v}[\tfrac12\|p_G(x)-e_y\|^2-\tfrac12\|p_{G+0.1\Delta_i}(x)-e_y\|^2].$$

Each loss lies in $[0,1]$ and each improvement in $[-1,1]$. Positive improvement means lower loss on that validator's local examples. The three fixed rules take, respectively, the median, mean of the three largest, or fourth-largest improvement among the nine other clients. Acceptance requires a score strictly greater than zero. The top-three control can retain minority support but is exposed to collusion; the fourth-largest control tests the cost of requiring support beyond three reports. No rule is selected after seeing results.

Each client has separate 128-image **report** and 128-image **audit** sets. Report data generates inputs to the scorer; audit data measures utility afterwards and never affects acceptance. Report and audit banks are globally disjoint, including across regimes and seeds, and exclude the checkpoint-training selections, server reference and global development data. Samples within the same role can recur across cases; those cases are not independent replications. Honest changes are represented in clients' current legitimate validation domains. Missing classes remain explicit in class-wise diagnostics.

The four report profiles are truthful reporting, directional extreme values, sign reversal, and evaluation with corrupted labels. Malicious updates and dishonest validation reports are different behaviours. An update can be honestly trained even when its author sends dishonest reports about others; the intent metric describes the update, not every action of its author. Cohorts and active windows are declared by the protocol, not inferred by a scorer.

Class-wise and overall cross-entropy changes are retained as secondary observations. FLShield motivates class-wise loss-impact measurement, but validates representative mixtures and applies a different filtering procedure. These individual-update, self-excluded Brier controls are our diagnostic adaptation. [FLShield primary paper](https://arxiv.org/html/2308.05832)."""

REPAIR_AND_ENVELOPE = """## Repair and arbitrary-report sensitivity

The attack-stops case ceases label poisoning and dishonest reporting after round 10. The sampled repaired checkpoints are rounds 12 and 20. They start from the same clean broadcasts as the corresponding clean cases: restored scores are therefore mechanically expected to match when all other inputs match. This checks score-level readmission and implementation consistency, not recovery speed, learned trust rehabilitation or recovery of an attacked global trajectory.

The arbitrary-report envelope is separate from the four concrete profiles. It examines how much a score could change when any three peer reports are replaced; an actively poisoned candidate already consumes one of the three compromised-client identities, leaving two compromised peers. The envelope must not be reported as measured attacker performance or as another independently fitted defence."""

LIMITS = """## Interpretation and reproduction

Honest intent and measured usefulness are separate labels. An honest update can increase loss on a validator from another domain, and a malicious update can improve some examples. Honest-minority and other-domain audit measurements therefore accompany the intent-based gate. A majority vote is not an intent oracle: after self-exclusion, a three-client drift cohort provides only two other same-domain validators, while coordinated peers can fabricate apparent support.

These correlated cases use fixed MNIST partitions, ten clients, a small candidate step, sparse checkpoints and previously used seeds. There is no new end-to-end global-model accuracy result. Passing a fixed-rule screen would justify a separately declared aggregation study; it would not prove robust FL deployment or novelty. Failing it identifies a limitation of these observations and rules, not a universal impossibility result.

Data and labels stay within simulated client-local evaluation calls. The experiment enforces a software boundary, not a production distributed system, cryptographic privacy protection or confidentiality guarantee for exchanged models. Local audit evaluation is research overhead; it is not an input available to a deployable scorer.

After the declared study completes, reproduce this saved-artifact presentation with:

```sh
Experiment/.venv/bin/python Experiment/code/client_validation_notebook.py
```

The builder checks completed artifacts and executes presentation cells only. It never launches local training, model evaluation, report manipulation or threshold selection."""


def make_notebook(study=DEFAULT_STUDY):
    """Create unexecuted presentation cells without opening any study artifact."""
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    setup = f"""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
CODE = Path({json.dumps(str(ROOT / 'code'))})
sys.path.insert(0,str(CODE))
from client_validation_study import verify, checked_result
from experiment_core import file_sha256
ROOT = CODE.parent
STUDY = Path({json.dumps(str(Path(study).resolve()))})
A = STUDY/'analysis'
STATUS = json.loads((STUDY/'status.json').read_text())
assert STATUS['status']=='complete' and STATUS['cases']==96 and STATUS['local_trainings']==960
P = verify(STUDY)
MANIFEST = json.loads((STUDY/'result_manifest.json').read_text())
assert MANIFEST['official_test_used'] is False and MANIFEST['global_models_trained']==0
assert MANIFEST['trusted_server_images']==0 and MANIFEST['local_trainings']==960
records = MANIFEST['records']
assert len(records)==96
META = []
for record in records:
    checked_result(record)
    metadata = json.loads((ROOT/record['run_path']/'metadata.json').read_text())
    assert metadata['official_test_used'] is False and metadata['global_model_advanced'] is False
    assert metadata['server_reference_forward_examples']==0 and metadata['global_development_forward_examples']==0
    META.append(metadata)
ACCESS = json.loads((STUDY/'access_audit.json').read_text())
assert ACCESS['cases']==96 and ACCESS['local_trainings']==960 and ACCESS['exact_prior_update_replays']==912
assert ACCESS['report_forward_examples']==ACCESS['audit_forward_examples']==1351680
assert ACCESS['trusted_server_images']==ACCESS['global_development_forward_examples']==ACCESS['global_models_advanced']==0
assert ACCESS['official_test_used'] is False and len(ACCESS['cases_audit'])==96
S = json.loads((A/'summary.json').read_text())
assert S['analysis_source_sha256']==file_sha256(CODE/'client_validation_analysis.py')
assert S['counts']['cases']==96 and S['counts']['candidate_updates']==960
D = pd.read_csv(A/'decision_rows.csv')
R = pd.read_csv(A/'grouped_rates.csv')
U = pd.read_csv(A/'audit_utility.csv')
E = pd.read_csv(A/'fault_envelopes.csv')
ER = pd.read_csv(A/'envelope_rates.csv')
CC = pd.read_csv(A/'class_coverage.csv')
assert len(D)==S['counts']['candidate_rule_profile_decisions']==11520
assert len(E)==S['counts']['fault_envelopes']==2880
RULES = ['peer_median','peer_top3_mean','peer_fourth_highest']
PROFILES = ['truthful','directional_extreme','sign_reversal','corrupted_labels']
REGIMES = ['IID','non-IID']; SEEDS = [20261014,20261015]; ROUNDS = [5,8,12,20]
SCENARIOS = ['clean','abrupt_shot_noise','gradual_stripe','label_flip','attack_stops','mixed_stripe_attack']
RULE_LABELS = {{'peer_median':'Median of peers','peer_top3_mean':'Mean of top 3 peers','peer_fourth_highest':'Fourth-highest peer'}}
PROFILE_LABELS = {{'truthful':'Truthful reports','directional_extreme':'Fabricated extremes','sign_reversal':'Reversed gains','corrupted_labels':'Flipped labels'}}
SCENARIO_LABELS = {{'abrupt_shot_noise':'Honest noise','gradual_stripe':'Honest stripes','mixed_stripe_attack':'Stripes with label attack'}}
METRIC_LABELS = {{'honest_update_retention':'Honest-update retention','poison_rejection':'Poisoned-update rejection','repaired_readmission':'Repaired-update readmission'}}
CASE_KEYS = ['regime','seed','scenario','round']
CANDIDATE_KEYS = CASE_KEYS+['candidate_id']
assert not D.duplicated(CANDIDATE_KEYS+['rule','profile']).any()
assert set(D.rule)==set(RULES) and set(D.profile)==set(PROFILES)
pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',16)
pd.set_option('display.width',180)
def percent(value):
    if pd.isna(value): return 'NA'
    return (f'{{100*value:.1f}}'.rstrip('0').rstrip('.')+'%')
def fraction(successes,total):
    return f'{{int(successes)}}/{{int(total)}} ({{percent(successes/total)}})' if total else '0/0 (NA)'
print('Verified 96 cases, 960 local trainings and 11,520 fixed-rule/profile decisions.')
print('No global model advanced; zero server-reference, global-development or official-test forward evaluation.')
"""
    cells = [
        md(INTRODUCTION),
        code(setup),
        code("""display(Markdown('**Primary fixed-profile gate: '+('PASS' if S['gate']['pass'] else 'NO GO')+'.**'))
if not S['gate']['pass']:
    display(Markdown('**No combined FedReTrust prototype was built: all three fixed rules failed the declared feasibility criteria.**'))
print('Passing fixed rules:',S['gate']['passing_rules'])
print('Predeclared tie-order selection, if any:',S['gate']['selected_rule'])
print('Separate arbitrary-peer envelope gate:', 'PASS' if S['envelope_gate']['pass'] else 'NO GO')
rows = []
for rule in RULES:
    empirical, envelope = S['gate']['rules'][rule], S['envelope_gate']['rules'][rule]
    rows.append({'rule':RULE_LABELS[rule],'primary_pass':empirical['pass'],
                 'failed_primary_groups':empirical['failed_groups'],'checked_primary_groups':empirical['groups'],
                 'envelope_pass':envelope['pass'],'failed_envelope_groups':envelope['failed_groups'],
                 'checked_envelope_groups':envelope['groups']})
display(pd.DataFrame(rows).set_index('rule'))
print(S['interpretation'])
decision_path = STUDY/'decision.json'
if decision_path.exists():
    decision = json.loads(decision_path.read_text())
    for name,digest in decision.get('evidence_sha256',{}).items():
        assert file_sha256(STUDY/name)==digest
    display(pd.Series({k:v for k,v in decision.items() if isinstance(v,(str,int,float,bool)) or v is None},name='Saved study decision').to_frame())
"""),
        md(METHOD),
        code("""display(pd.DataFrame([{'rule':RULE_LABELS[rule],'definition':S['protocol']['rule_definitions'][rule]} for rule in RULES]))
display(pd.DataFrame([{'profile':PROFILE_LABELS[profile],'definition':S['protocol']['profiles'][profile]} for profile in PROFILES]))
cohorts = S['protocol']['reporting_cohorts']
display(pd.Series(cohorts,name='Declared reporting actors').to_frame())
jobs = pd.DataFrame([record['job'] for record in records])
coverage = jobs.groupby(['regime','seed','scenario'])['round'].agg(cases='count',checkpoints=lambda values:','.join(map(str,sorted(values))))
display(coverage)
assert coverage.cases.eq(4).all()
print('Each case has ten candidates and ten report/audit validators. All scorers exclude the candidate’s own report.')
print('Images are sampled without class balancing. Report/audit roles are globally disjoint; within-role reuse remains.')
display(pd.Series({key:value for key,value in ACCESS.items() if isinstance(value,(int,float,bool))},name='Verified access and replay audit').to_frame())
print('912 previously observed updates replay exactly. The 48 mixed-scenario stripe-client updates have no identical earlier condition.')
"""),
        md("""## All fixed rules, reporting profiles and intent-based outcomes

Each rule must satisfy the entire declared grid: at least 95% retention for every honest-update subgroup, at least 80% rejection for poisoned updates, and at least 80% readmission for repaired updates. These criteria are applied separately for every partition, seed, scenario, group and reporting profile. One rule must pass all applicable conditions; a favourable pooled rate cannot rescue a failing subgroup.

The first table gives the worst subgroup rate for each metric. Subsequent tables retain every seed, scenario and group with numerator and denominator. An honest-intent restriction is an intent-based outcome; its utility meaning is examined separately on audit data."""),
        code("""worst = R.groupby(['rule','profile','metric']).success_rate.min().unstack('metric')
worst = worst.reindex(pd.MultiIndex.from_product([RULES,PROFILES],names=['rule','profile']))
display(worst.rename(columns=METRIC_LABELS).map(percent))
for profile in PROFILES:
    display(Markdown('**'+PROFILE_LABELS[profile]+' — every seed, scenario and update group**'))
    rows = R[R.profile==profile].copy()
    rows['observed'] = rows.apply(lambda row:fraction(row.successes,row.n_decisions),axis=1)
    rows['required'] = rows.required_rate.map(percent)
    table = rows.pivot(index=['regime','seed','scenario','group','metric','required'],columns='rule',values='observed')
    display(table.reindex(columns=RULES).rename(columns=RULE_LABELS))
print('These are correlated candidate/checkpoint decisions, not independent replication counts.')
"""),
        code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIGURES = A/'figures'; FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def export_figure(fig,stem):
    fig.savefig(FIGURES/(stem+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIGURES/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)
    display(Image(filename=str(FIGURES/(stem+'.png'))))
fig,axes = plt.subplots(3,4,figsize=(16,10))
for ri,rule in enumerate(RULES):
    for pi,profile in enumerate(PROFILES):
        ax = axes[ri,pi]
        rows = D[(D.rule==rule)&(D.profile==profile)]
        counts = np.array([[int((rows[rows.poisoned_update==poison].accepted==accept).sum())
                            for accept in (True,False)] for poison in (False,True)])
        totals = counts.sum(axis=1)
        rates = np.divide(counts,totals[:,None],out=np.full((2,2),np.nan),where=totals[:,None]!=0)
        ax.imshow(rates,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for y in range(2):
            for x in range(2):
                ax.text(x,y,f'{counts[y,x]}/{totals[y]}\\n{percent(rates[y,x])}',ha='center',va='center',
                        color='white' if rates[y,x]>.55 else 'black',fontsize=10)
        ax.set_xticks([0,1],['Accepted','Rejected'])
        ax.set_yticks([0,1],['Honest update','Poisoned update'])
        ax.set_title(RULE_LABELS[rule]+'\\n'+PROFILE_LABELS[profile],fontsize=10)
fig.suptitle('Peer-validation decisions: update type versus acceptance\\nCounts and row percentages pooled over cases; subgroup tests govern the decision')
fig.tight_layout(rect=(0,0,1,.93))
export_figure(fig,'client_validation_intent_confusion')
"""),
        md("""## Independent usefulness: the client's domain and other domains

Audit examples are disjoint from report examples and never enter a scoring rule or the primary gate. Positive audit gain means the small candidate step reduced Brier loss on that evaluation set; it is not proof of benign intent. The tables distinguish usefulness to the candidate's own client, the changed-domain cohort, all domains and the remaining domains.

The utility-conditioned retention table asks two separate questions: did a rule retain an honest update that helped its own audit set, and did it retain a changed client's update that helped the changed-domain cohort? A zero denominator is undefined, not 0% retention. These are descriptive subgroup checks, not additional tuned gates."""),
        code("""audit_columns = ['self_audit_gain','all_domains_audit_gain','changed_cohort_audit_gain','remaining_domains_audit_gain']
BASE = D[(D.rule==RULES[0])&(D.profile=='truthful')].copy()
assert len(BASE)==960 and not BASE.duplicated(CANDIDATE_KEYS).any()
domain_means = BASE.groupby(['regime','seed','scenario','group'])[audit_columns].mean()
domain_means['candidate_count'] = BASE.groupby(['regime','seed','scenario','group']).size()
display(domain_means.round(6))
for criterion,title in [('positive_self_audit_gain','Honest updates useful to their own client'),
                        ('positive_changed_cohort_audit_gain','Changed-client updates useful to the changed cohort')]:
    rows = U[U.audit_criterion==criterion].copy()
    if criterion=='positive_changed_cohort_audit_gain': rows = rows[rows.group=='changed']
    totals = rows.groupby(['regime','seed','rule','profile'])[['n_audit_useful','useful_accepted']].sum().reset_index()
    totals['observed'] = totals.apply(lambda row:fraction(row.useful_accepted,row.n_audit_useful),axis=1)
    display(Markdown('**'+title+' — acceptance among useful candidates, pooled across relevant scenarios**'))
    display(totals.pivot(index=['regime','seed','profile'],columns='rule',values='observed').reindex(columns=RULES).rename(columns=RULE_LABELS))
print('Scenario-specific useful-retention rates and denominators remain in',A/'audit_utility.csv')
"""),
        code("""fig,axes = plt.subplots(1,2,figsize=(12,5.5),sharex=True,sharey=True)
scenario_colors = {'abrupt_shot_noise':'#0072B2','gradual_stripe':'#D55E00','mixed_stripe_attack':'#009E73'}
for ax,regime in zip(axes,REGIMES):
    changed = BASE[(BASE.regime==regime)&(BASE.group=='changed')]
    for scenario,color in scenario_colors.items():
        rows = changed[changed.scenario==scenario]
        ax.scatter(rows.changed_cohort_audit_gain,rows.remaining_domains_audit_gain,
                   color=color,s=25,alpha=.65,label=SCENARIO_LABELS[scenario])
    ax.axhline(0,color='#666666',lw=1); ax.axvline(0,color='#666666',lw=1)
    ax.set_title(regime); ax.set_xlabel('Loss improvement on clients with changed data (Brier)')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Loss improvement on remaining clients (Brier)')
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Honest updates after a data change: who benefits on independent audit data?')
fig.tight_layout(rect=(0,0,1,.94))
export_figure(fig,'client_validation_domain_utility')
"""),
        md("""## Class coverage and secondary cross-entropy observations

The Brier rule uses the validator's unbalanced 128-example local distribution. It does not assume that all classes are present. Coverage is therefore part of interpreting a report, especially under non-IID data. The table below counts unique client holdouts once per role, seed and partition, rather than inflating coverage by counting reused checkpoints or scenarios.

Class-wise cross-entropy and Brier changes are preserved as secondary diagnostics. They are never used to choose a rule, set a threshold or replace a failed Brier gate."""),
        code("""coverage_keys = ['regime','seed','validator_id','role']
class_columns = [f'class_{digit}_count' for digit in range(10)]
assert CC.groupby(coverage_keys)[class_columns].nunique().le(1).all().all()
unique_coverage = CC.drop_duplicates(coverage_keys).copy()
assert len(unique_coverage)==80
coverage_rows = []
for (regime,seed,role),rows in unique_coverage.groupby(['regime','seed','role']):
    counts = rows[class_columns].to_numpy()
    coverage_rows.append({'regime':regime,'seed':seed,'role':role,'validators':len(rows),
                          'examples':int(rows.examples.sum()),'minimum_classes_per_validator':int(rows.observed_classes.min()),
                          'mean_classes_per_validator':float(rows.observed_classes.mean()),
                          'missing_validator_class_cells':int((counts==0).sum()),
                          'validator_class_cells_below_10_examples':int((counts<10).sum())})
display(pd.DataFrame(coverage_rows).set_index(['regime','seed','role']).round(2))
present = unique_coverage.copy()
present[class_columns] = present[class_columns].gt(0).astype(int)
coverage_grid = present.groupby(['regime','seed','role'])[class_columns].sum()
fig,ax = plt.subplots(figsize=(12,5.5))
values = coverage_grid.to_numpy()
picture = ax.imshow(values,vmin=0,vmax=10,cmap='Blues',aspect='auto')
for y in range(len(coverage_grid)):
    for x in range(10):
        ax.text(x,y,str(values[y,x]),ha='center',va='center',color='white' if values[y,x]>=6 else 'black')
ax.set_xticks(range(10),range(10)); ax.set_xlabel('Digit class')
ax.set_yticks(range(len(coverage_grid)),[f'{regime} · {seed} · {role.title()}' for regime,seed,role in coverage_grid.index])
ax.set_title('How many clients have examples of each class?')
fig.colorbar(picture,ax=ax,label='Clients with at least one example (out of 10)')
fig.tight_layout()
export_figure(fig,'client_validation_class_coverage')
print('Class coverage counts all ten clients; excluding a candidate’s own report can reduce its available peer coverage by one.')
label_by_candidate = BASE.set_index(CANDIDATE_KEYS)['group'].to_dict()
secondary = []
for record in records:
    job = record['job']
    for role,filename in [('report','reports.json'),('audit','audit.json')]:
        for row in json.loads((ROOT/record['run_path']/filename).read_text()):
            key = tuple(job[name] for name in CASE_KEYS)+(row['candidate_id'],)
            secondary.append({**job,'role':role,'candidate_id':row['candidate_id'],'validator_id':row['validator_id'],
                              'group':label_by_candidate[key],
                              'brier_gain':row['mean_brier_gain'],'cross_entropy_gain':row['mean_cross_entropy_gain'],
                              'defined_classes':sum(value is not None for value in row['class_cross_entropy_gains'])})
SECONDARY = pd.DataFrame(secondary)
display(SECONDARY.groupby(['regime','seed','scenario','group','role'])[['brier_gain','cross_entropy_gain','defined_classes']].mean().round(6))
print('Secondary gains use correct-target summaries; flipped-label reports are a separate predeclared attack profile.')
print('Full per-class arrays remain in each verified reports.json/audit.json; no class reweighting or imputation was applied.')
"""),
        md(REPAIR_AND_ENVELOPE),
        code("""print('Repaired candidate observations:',S['repair_equivalence']['n_repaired_observations'])
print('Exact clean-counterfactual equivalence:',S['repair_equivalence']['all_equal'])
print(S['repair_equivalence']['interpretation'])
repair = R[R.group=='repaired'].copy()
repair['observed'] = repair.apply(lambda row:fraction(row.successes,row.n_decisions),axis=1)
display(repair.pivot(index=['regime','seed','profile'],columns='rule',values='observed').reindex(columns=RULES).rename(columns=RULE_LABELS))
display(Markdown('**Separate candidate-specific arbitrary-peer envelope**'))
display(pd.DataFrame([{'rule':RULE_LABELS[rule],'pass':S['envelope_gate']['rules'][rule]['pass'],
                       'failed_groups':S['envelope_gate']['rules'][rule]['failed_groups'],
                       'checked_groups':S['envelope_gate']['rules'][rule]['groups']} for rule in RULES]))
envelope = ER.copy()
envelope['observed'] = envelope.apply(lambda row:fraction(row.successes,row.n_decisions),axis=1)
display(envelope.pivot(index=['regime','seed','scenario','group','metric'],columns='rule',values='observed').reindex(columns=RULES).rename(columns=RULE_LABELS))
print('The envelope can use different adversarial peers for different candidates; it is not one simultaneously executed report attack.')
print('Its hypothetical fault budget also differs from the repaired concrete profile, where all declared misconduct has stopped.')
"""),
        md("""## Measured compute and hypothetical communication payload

All four report profiles reuse saved predictions and reports; they do not cause four sets of model forwards. Report validation is potential mechanism work. Audit validation is research-only evaluation. Summed per-case wall times are not elapsed study time when workers overlap.

For a simple deployment, each of ten validators would receive the nine other candidate models: 90 model payloads per case. Existing broadcast delivery and ordinary training uploads are excluded from this additional validation payload. Scalar-report arithmetic below assumes one 64-bit gain per candidate–peer pair and excludes IDs, protocol framing, encryption and secondary diagnostics. These are payload estimates, not measured network traffic."""),
        code("""cost_rows = []
for record,metadata in zip(records,META):
    costs = metadata['local_validation_costs']
    cost_rows.append({**record['job'],'local_trainings':record['local_trainings'],
                      'training_seconds':metadata['training_seconds'],
                      'validation_seconds':sum(item['elapsed_seconds'] for item in costs),
                      'report_forward_examples':sum(item['models_per_role']*item['report_examples'] for item in costs),
                      'audit_forward_examples':sum(item['models_per_role']*item['audit_examples'] for item in costs),
                      'total_forward_examples':sum(item['total_forward_examples'] for item in costs),
                      'case_wall_seconds':record['worker_wall_seconds']})
COST = pd.DataFrame(cost_rows)
assert COST.total_forward_examples.sum()==2703360
assert COST.local_trainings.sum()==960
assert COST.report_forward_examples.sum()==ACCESS['report_forward_examples']
assert COST.audit_forward_examples.sum()==ACCESS['audit_forward_examples']
totals = COST[['local_trainings','training_seconds','validation_seconds','report_forward_examples','audit_forward_examples','total_forward_examples','case_wall_seconds']].sum()
display(totals.rename('Sum over 96 cases').to_frame())
display(COST.groupby(['regime','seed'])[['training_seconds','validation_seconds','case_wall_seconds']].median().rename(columns=lambda name:'median_'+name).round(3))
model_sizes = {metadata['float32_model_payload_bytes'] for metadata in META}
assert len(model_sizes)==1
model_bytes = model_sizes.pop()
assert sum(row['naive_model_distribution_bytes'] for row in ACCESS['cases_audit'])==96*90*model_bytes
assert sum(row['peer_scalar_float64_payload_bytes'] for row in ACCESS['cases_audit'])==96*90*8
payload = pd.DataFrame([
    {'quantity':'One float32 model','per_case_bytes':model_bytes,'study_bytes':np.nan},
    {'quantity':'90 peer-model payloads','per_case_bytes':90*model_bytes,'study_bytes':96*90*model_bytes},
    {'quantity':'90 float64 gain reports (assumed encoding)','per_case_bytes':90*8,'study_bytes':96*90*8},
]).set_index('quantity')
display(payload)
print('Naive peer-model payload across all cases:',round(96*90*model_bytes/2**30,3),'GiB; not measured traffic.')
print('Stored JSON diagnostics and independent audit work are not included in the hypothetical minimal scalar-report payload.')
COST.to_csv(A/'notebook_case_costs.csv',index=False)
print('Figures exported as PNG/PDF:',FIGURES)
"""),
        md(LIMITS),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name":"FedReTrust Experiment (Python 3.14)",
                                    "language":"python","name":"fedretrust-experiment"}
    return notebook


def _execute(notebook, output):
    """Use the project's socket-free IPython workflow for presentation cells."""
    os.environ.setdefault("IPYTHONDIR", "/tmp/fedretrust_ipython")
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    previous = Path.cwd()
    os.chdir(ROOT / "code")
    try:
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            cell.execution_count = count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbf.v4.new_output("stream", name="stderr", text=captured.stderr))
            for item in captured.outputs:
                cell.outputs.append(nbf.v4.new_output("display_data", data=item.data, metadata=item.metadata))
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f"Notebook cell {count}: {result.error_before_exec or result.error_in_exec}")
    finally:
        os.chdir(previous)
    notebook.metadata["execution_note"] = (
        "All presentation cells executed with IPython InteractiveShell after completed-study verification. "
        "Saved artifacts only; no local training or model evaluation.")
    nbf.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    return output


def build(study=DEFAULT_STUDY, output=DEFAULT_NOTEBOOK, execute=True):
    study, output = Path(study).resolve(), Path(output).resolve()
    status = json.loads((study/'status.json').read_text())
    if status.get('status')!='complete' or status.get('cases')!=96 or status.get('local_trainings')!=960:
        raise ValueError('Complete all 96 cases and their declared analysis before creating this review.')
    for name in ('protocol.json','result_manifest.json','access_audit.json','analysis/summary.json',
                 'analysis/decision_rows.csv','analysis/grouped_rates.csv','analysis/audit_utility.csv',
                 'analysis/fault_envelopes.csv','analysis/envelope_rates.csv','analysis/class_coverage.csv'):
        if not (study/name).is_file():
            raise ValueError(f'Missing completed review artifact: {name}')
    from client_validation_study import verify
    verify(study)
    notebook = make_notebook(study)
    if execute:
        return _execute(notebook,output)
    nbf.validate(notebook)
    output.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,output)
    return output


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    parser.add_argument('--output',type=Path,default=DEFAULT_NOTEBOOK)
    parser.add_argument('--no-execute',action='store_true')
    args = parser.parse_args()
    print(build(args.study,args.output,execute=not args.no_execute))
