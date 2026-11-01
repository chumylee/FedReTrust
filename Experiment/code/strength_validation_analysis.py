"""Predeclared scale curves and report-only weight-selection controls."""
from collections import defaultdict
import copy
import itertools
import json
import math
from pathlib import Path
import statistics
import client_validation_analysis as base
from client_validation_analysis import ROOT, RULES, PROFILES, REGIMES, SEEDS, ROUNDS, SCENARIOS
from strength_validation_local import SCALES, SEMANTICS, verify_endpoint_rows

NONZERO = (.25, .5, 1.)
ADAPTIVE_PROFILES = (*PROFILES, 'force_full')
BENEFIT_RETAINED = .75
HARM_RETAINED = .5
UTILITY_SUCCESS = .8
EPS = 1e-12
FIXED_CONTROL_PROTOCOL = copy.deepcopy(base.DIAGNOSTIC_PROTOCOL)
FIXED_CONTROL_PROTOCOL["candidate_model"] = SEMANTICS["model"]
FIXED_CONTROL_PROTOCOL["reports"]["metric"] = "Mean Brier loss of omitted-client model minus mean Brier loss at the chosen fixed strength; positive is improvement."
FIXED_CONTROL_PROTOCOL["scope"] = "Each fixed-strength comparison uses the same pure scoring rules, actor profiles and intent criteria as the previous pilots, within the aggregate context defined here."
DIAGNOSTIC_PROTOCOL = {
    'version': 'controlled_contribution_strength_v1', 'scales': list(SCALES),
    'model': SEMANTICS, 'fixed_controls': FIXED_CONTROL_PROTOCOL,
    'adaptive': {
        'rules': list(RULES), 'profiles': list(ADAPTIVE_PROFILES),
        'decision': 'For each candidate choose the nonzero scale with largest strictly positive peer score. Exact ties prefer the smaller scale. If none is positive, choose zero. Zero has known score zero, not a client-reported score.',
        'interface': 'The selector receives a map of three scalar scores only; no audit, labels, scenario, actor identities, classes or trusted data.',
        'force_full_profile': 'Same three bad validators as previous profiles. They report +1 only for actively poisoned candidates at scale 1, and -1 at every other nonzero candidate/scale. Correct peers retain their true scale-specific report.',
        'scope': 'Offline predeclared weighting controls, not a new FedReTrust training implementation. Each selected utility is still a one-client counterfactual with the other nine at full strength.',
    },
    'utility_tradeoff': {
        'benefit': 'D(s) is mean correct-label independent audit Brier gain across honestly changed domains.',
        'harm': 'H(s)=max(0,-mean audit Brier gain across remaining domains). Both are relative to omission of the candidate, not to the common broadcast.',
        'criterion': 'D(s)>1e-12; D(s)>=0.75*max(D(1),0)-1e-12; H(s)<=0.5*H(1)+1e-12.',
        'benefit_retained': BENEFIT_RETAINED, 'harm_retained': HARM_RETAINED,
        'nonpositive_full_benefit': 'A scale must produce strictly positive benefit; preserving a negative full-strength effect does not count.',
        'already_no_collateral_harm': 'Full strength may qualify when it is beneficial and has no collateral harm; fractional-only gains are reported separately.',
        'feasibility_envelope': 'Offline, audit-aware best-existence check over the three nonzero scales. It is an optimistic bound, not a deployable oracle policy.',
        'required_rate': UTILITY_SUCCESS,
        'grouping': 'At least 80% of honestly changed candidates in every seed, regime and changed scenario separately (12 groups). Selection controls must also achieve this in every reporting profile (60 groups per rule).',
        'audit_boundary': 'Audit summaries assess scientific feasibility and the research continuation gate. They never enter score computation or scale selection. These reused development audits are not an untouched confirmation set.',
    },
    'continuation': 'Only continue to a separately declared end-to-end pilot if the feasibility envelope passes all 12 groups and a fixed selector passes both the unchanged intent criteria (95% honest,80% poison rejection,80% repaired) under all five profiles and the utility trade-off gate in all 60 groups. First passing rule in fixed order. Otherwise stop without fitting another score or threshold.',
    'report_envelopes': 'Keep the earlier exact per-scale any-client 3-fault envelope (poisoned candidate leaves 2 lying peers). These separate per-scale bounds are not an exact joint adversarial scale-selection guarantee.',
    'scope': '96 cases, two reused seeds, fixed partitions and four clean checkpoints; correlated exploratory observations. No official test, global development feedback or trusted server images. No selected combination or global trajectory evaluated in this stage.',
}


def select_scale(scores):
    """Report-only pure selector. Zero cannot be made attractive by a fake report."""
    if set(scores) != set(NONZERO): raise ValueError('Require all three nonzero scale scores.')
    for value in scores.values(): base._bounded(value)
    positive = [s for s in NONZERO if scores[s] > 0]
    return max(positive, key=lambda s:(scores[s], -s)) if positive else 0.


def tradeoff(benefit, remaining_gain, full_benefit, full_remaining_gain):
    values = (benefit, remaining_gain, full_benefit, full_remaining_gain)
    if any(v is None or not math.isfinite(v) for v in values): raise ValueError('Finite domain gains required.')
    harm, full_harm = max(0., -remaining_gain), max(0., -full_remaining_gain)
    return (benefit > EPS and benefit >= BENEFIT_RETAINED*max(full_benefit,0.)-EPS
            and harm <= HARM_RETAINED*full_harm+EPS)


def profile_matrix(case, profile):
    ids = list(range(10)); j = case['job']
    bad = base._reporting_cohort(j['scenario'], j['round'])
    if profile != 'force_full':
        return base.profile_reports(case['reports'],case['flipped'],ids,ids,profile,bad,case['poisoned'])
    result = copy.deepcopy(case['reports'])
    for cid in ids:
        for vid in bad:
            result[cid][vid] = 1. if cid in case['poisoned'] and j['scale']==1. else -1.
    return result


def _job_key(job): return tuple(job[n] for n in ('regime','seed','scenario','round'))


def load_cases(study):
    manifest = json.loads((study/'result_manifest.json').read_text())
    protocol = json.loads((study/'protocol.json').read_text())
    expected = set(itertools.product(REGIMES, SEEDS, SCENARIOS, ROUNDS))
    seen, cases, inputs = set(), {s:[] for s in SCALES}, []
    for record in manifest['records']:
        j=record['job']; key=_job_key(j)
        if key not in expected or key in seen or record['status']!='complete': raise ValueError('Invalid case grid.')
        if record['source_sha256'] != protocol['source_sha256']: raise ValueError('Source fingerprint mismatch.')
        seen.add(key); path=ROOT/record['run_path']
        required={'reports.json','audit.json','metadata.json','training.json','replay_audit.json'}
        if not required.issubset(record['artifact_sha256']): raise ValueError('Missing artifact hashes.')
        for name,digest in record['artifact_sha256'].items():
            if base._hash(path/name)!=digest: raise ValueError('Case artifact changed.')
        meta=json.loads((path/'metadata.json').read_text())
        if meta['job']!=j or meta['scenario']['name']!=j['scenario']: raise ValueError('Case metadata mismatch.')
        if meta['official_test_used'] or meta['global_model_advanced'] or meta['server_reference_forward_examples'] or meta['global_development_forward_examples']: raise ValueError('Disallowed evaluation access.')
        raw={role:json.loads((path/(role+'.json')).read_text()) for role in ('reports','audit')}
        for role in raw:
            if len(raw[role])!=400 or set(r['scale'] for r in raw[role])!=set(SCALES): raise ValueError('Incomplete strength curve.')
        for s in SCALES:
            matrices={}
            for role,field in (('reports','mean_brier_gain'),('audit','mean_brier_gain'),('flipped','flipped_mean_brier_gain')):
                rows=[r for r in raw['reports' if role=='flipped' else role] if r['scale']==s]
                matrix,index=base._matrix_rows(rows,field); matrices[role]=matrix
                for row in rows:
                    if row['examples']!=128 or sum(row['class_counts'])!=128: raise ValueError('Local coverage mismatch.')
                    prefix='flipped_' if role=='flipped' else ''
                    difference=row[prefix+'without_client_mean_brier']-row[prefix+'scaled_mean_brier']
                    if not math.isclose(row[field],difference,abs_tol=1e-12,rel_tol=0): raise ValueError('Inconsistent contribution gain.')
            if s==0. and any(v!=0. for mat in matrices.values() for row in mat for v in row): raise ValueError('Nonzero zero-control gain.')
            poison,changed,repaired,groups=base._labels(meta['scenario'],j['round'])
            cases[s].append({'job':{**j,'scale':s},'metadata':meta,**matrices,'poisoned':poison,'changed':changed,'repaired':repaired,'groups':groups})
        inputs.append({'job':j,'run_path':str(path),'artifact_sha256':record['artifact_sha256']})
    if seen!=expected: raise ValueError('Incomplete 96-case grid.')
    return cases,inputs


def adaptive_rows(cases):
    indexed={s:{_job_key(c['job']):c for c in group} for s,group in cases.items()}
    rows=[]; ids=list(range(10))
    for key,full in indexed[1.].items():
        for profile in ADAPTIVE_PROFILES:
            matrices={s:profile_matrix(indexed[s][key],profile) for s in NONZERO}
            for rule in RULES:
                scores={s:base.score_reports(matrices[s],ids,ids,rule) for s in NONZERO}
                for cid in ids:
                    chosen=select_scale({s:scores[s][cid] for s in NONZERO})
                    measurements=base._audit_measures(indexed[chosen][key],cid)
                    full_measure=base._audit_measures(full,cid)
                    good=(tradeoff(measurements['changed_cohort_audit_gain'],measurements['remaining_domains_audit_gain'],
                                   full_measure['changed_cohort_audit_gain'],full_measure['remaining_domains_audit_gain'])
                          if full['groups'][cid]=='changed' else None)
                    rows.append({**{k:v for k,v in full['job'].items() if k!='scale'},'candidate_id':cid,
                        'rule':rule,'profile':profile,'group':full['groups'][cid],
                        'poisoned_update':cid in full['poisoned'],'selected_scale':chosen,
                        'effective_client_coefficient':.1*chosen,'accepted':chosen>0,
                        'score':scores[chosen][cid] if chosen else 0.,
                        'score_025':scores[.25][cid],'score_050':scores[.5][cid],'score_100':scores[1.][cid],
                        **measurements,'utility_tradeoff_met':good})
    return rows


def curve_rows(cases):
    full={_job_key(c['job']):c for c in cases[1.]}; rows=[]
    for s,group in cases.items():
        for case in group:
            for cid in range(10):
                a=base._audit_measures(case,cid); f=base._audit_measures(full[_job_key(case['job'])],cid)
                good=tradeoff(a['changed_cohort_audit_gain'],a['remaining_domains_audit_gain'],
                              f['changed_cohort_audit_gain'],f['remaining_domains_audit_gain']) if case['groups'][cid]=='changed' else None
                rows.append({**case['job'],'candidate_id':cid,'group':case['groups'][cid],
                             'poisoned_update':cid in case['poisoned'],**a,
                             'full_changed_cohort_audit_gain':f['changed_cohort_audit_gain'],
                             'full_remaining_domains_audit_gain':f['remaining_domains_audit_gain'],
                             'remaining_domain_harm':max(0.,-a['remaining_domains_audit_gain']),
                             'utility_tradeoff_met':good})
    return rows


def feasibility_rows(curves):
    grouped=defaultdict(list)
    for r in curves:
        if r['group']=='changed': grouped[(*_job_key(r),r['candidate_id'])].append(r)
    rows=[]
    for members in grouped.values():
        first=members[0]; valid=[r['scale'] for r in members if r['utility_tradeoff_met']]
        rows.append({**{k:first[k] for k in ('regime','seed','scenario','round','candidate_id')},
                     'any_scale_feasible':bool(valid),'fractional_scale_feasible':any(s in (.25,.5) for s in valid),
                     'smallest_feasible_scale':min(valid) if valid else None,
                     'full_strength_feasible':1. in valid,
                     'audit_aware_upper_bound_only':True})
    return rows


def utility_rates(rows,adaptive=False):
    keys=('rule','profile','regime','seed','scenario') if adaptive else ('regime','seed','scenario')
    groups=defaultdict(list)
    for r in rows:
        if adaptive and r['group']!='changed':continue
        groups[tuple(r[k] for k in keys)].append(r)
    output=[]
    for key,members in sorted(groups.items()):
        successes=sum(bool(r['utility_tradeoff_met' if adaptive else 'any_scale_feasible']) for r in members)
        output.append({**dict(zip(keys,key)),'n_candidates':len(members),'successes':successes,
                       'success_rate':successes/len(members),'required_rate':UTILITY_SUCCESS,
                       'pass':successes*5>=len(members)*4})
    return output


def analyse_study(study_path):
    study=Path(study_path); cases,inputs=load_cases(study)
    fixed, envelopes, rates, envelope_rates=[],[],[],[]
    gates, envelope_gates, repairs={},{},{}
    for scale in NONZERO:
        d,e=base.decision_rows(cases[scale]); r=base.grouped_rates(d); er=base.grouped_rates(e)
        for row in r+er:row['scale']=scale
        fixed+=d; envelopes+=e; rates+=r; envelope_rates+=er
        gates[str(scale)]=base.gate_summary(r)
        envelope_gates[str(scale)]=base.gate_summary(er,('any_peer_fault_envelope',))
        repairs[str(scale)]=base.repair_equivalence(cases[scale],d)
    adaptive=adaptive_rows(cases); ar=base.grouped_rates(adaptive)
    adaptive_gate=base.gate_summary(ar,ADAPTIVE_PROFILES)
    curves=curve_rows(cases); feasible=feasibility_rows(curves)
    fr=utility_rates(feasible); ur=utility_rates(adaptive,True)
    expected={(r,s,c) for r in REGIMES for s in SEEDS for c in ('abrupt_shot_noise','gradual_stripe','mixed_stripe_attack')}
    if {(r['regime'],r['seed'],r['scenario']) for r in fr}!=expected or len(fr)!=12:raise ValueError('Incomplete feasibility gate.')
    u_gates={}
    for rule in RULES:
        group=[r for r in ur if r['rule']==rule]
        if len(group)!=60:raise ValueError('Incomplete adaptive utility gate.')
        u_gates[rule]={'pass':all(r['pass'] for r in group),'groups':len(group),
                       'failed_groups':sum(not r['pass'] for r in group)}
    feasibility_pass=all(r['pass'] for r in fr)
    passing=[rule for rule in RULES if feasibility_pass and adaptive_gate['rules'][rule]['pass'] and u_gates[rule]['pass']]
    summary={'protocol':DIAGNOSTIC_PROTOCOL,'analysis_source_sha256':base._hash(__file__),'inputs':inputs,
        'counts':{'cases':96,'local_trainings':960,'candidate_scale_observations':len(curves),
                  'fixed_rule_profile_decisions':len(fixed),'adaptive_rule_profile_decisions':len(adaptive),
                  'fixed_fault_envelopes':len(envelopes)},
        'fixed_gates':gates,'fixed_envelope_gates':envelope_gates,'repair_equivalence':repairs,
        'adaptive_intent_gate':adaptive_gate,'adaptive_utility_gates':u_gates,
        'feasibility_gate':{'pass':feasibility_pass,'groups':12,'failed_groups':sum(not r['pass'] for r in fr),
                            'meaning':'Audit-aware existence bound, not a deployed selection rule.'},
        'continuation':{'pass':bool(passing),'passing_rules':passing,'selected_rule':passing[0] if passing else None,
                        'decision':'declare_end_to_end_followup' if passing else 'stop_this_weighting_combination'},
        'interpretation':'All outcomes are one-client-at-a-time contribution counterfactuals. Choosing several scales simultaneously changes their contexts. A passing screen would still require an actual combined-model experiment. Audit-aware feasibility is never a report-only policy.'}
    output=study/'analysis';output.mkdir(exist_ok=True)
    tables={'fixed_decisions':fixed,'fixed_grouped_rates':rates,'fixed_fault_envelopes':envelopes,
            'fixed_envelope_rates':envelope_rates,'adaptive_decisions':adaptive,'adaptive_grouped_rates':ar,
            'audit_curves':curves,'feasibility_candidates':feasible,'feasibility_rates':fr,'adaptive_utility_rates':ur}
    for name,rows in tables.items():base._write_csv(output/(name+'.csv'),rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return summary
