"""Offline evaluation: paired seeds, legitimate tasks, censored episodes and costs.

This module may read simulator truth for evaluation. Nothing here feeds back
into an aggregator, training, candidate selection or final parameter settings.
"""
from pathlib import Path
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from experiment_core import file_sha256
from research_runs import ROOT, METHODS, PARTITIONS, verify_freeze
from research_study import DEFAULT_STUDY, save, now, result_for, task_series
from scenario_metrics import (longer_scenarios, matched_reference_name, validate_matched_runs,
                              paired_damage, classify_model_episode, classify_contribution_episode,
                              contribution_reduction_event, contribution_norm_ratios)


HONEST_SCENARIOS = ("abrupt_shot_noise","gradual_stripe","honest_change_stops","mixed_drift_and_attack")
ATTACK_SCENARIOS = ("label_flip","gaussian_update","attack_stops","recurring_attack","mixed_drift_and_attack")


def group_fields(keys,values):
    return {key:value.item() if isinstance(value,np.generic) else value for key,value in zip(keys,values)}


def method_label(job):
    if job.get("overrides"):
        return "FedReTrust:"+",".join(f"{k}={v:g}" for k,v in sorted(job["overrides"].items()))
    variant = job.get("variant","full")
    return job["method"] if variant == "full" else "FedReTrust:"+variant


def run_key(job, scenario=None):
    return job["regime"],job["seed"],method_label(job),scenario or job["scenario"]


def identity(result):
    job = result["job"]
    return {"phase":job["phase"],"regime":job["regime"],"seed":job["seed"],
            "method":method_label(job),"scenario":job["scenario"],"job_id":result["job_id"]}


def rounds_only(series):
    return {r:v for r,v in series.items() if r>0}


def mean_window(series,start,end):
    if any(r not in series for r in range(start,end+1)):
        raise ValueError("Incomplete measurement window.")
    return float(np.mean([series[r] for r in range(start,end+1)]))


def interval(values, expected_n=5):
    """Do not imply complete paired evidence when failed seeds are missing."""
    if expected_n != 5:
        raise ValueError("This predeclared analysis requires five seeds (Student-t df=4).")
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError("Invalid finite seed measurements.")
    n=len(values)
    if n>expected_n:
        raise ValueError("Too many observations; check for duplicate seed rows.")
    mean=float(values.mean()) if n else None
    sd=float(values.std(ddof=1)) if n>1 else None
    half=float(2.7764451051977987*sd/math.sqrt(n)) if n==expected_n else None
    return {"n":n,"expected_n":expected_n,"complete_seed_coverage":n==expected_n,
            "mean":mean,"sd":sd,"ci_low":mean-half if half is not None else None,
            "ci_high":mean+half if half is not None else None}


def rename_episode_flag(value):
    value=dict(value)
    # SV-01: the older API flag includes continuing impairment, not only a new onset.
    value["any_post_event_impairment_observed"]=value.pop("delayed_post_event_impairment_observed")
    return value


def checked_pair(observed,reference):
    for result in (observed,reference):
        if result["status"]!="complete":
            raise ValueError("A matched comparison requires two completed runs.")
    left=json.loads((ROOT/observed["run_path"]/"run.json").read_text())
    right=json.loads((ROOT/reference["run_path"]/"run.json").read_text())
    return validate_matched_runs(left,right)


def cross_method_pair(observed,reference):
    left=json.loads((ROOT/observed["run_path"]/"run.json").read_text())
    right=json.loads((ROOT/reference["run_path"]/"run.json").read_text())
    keys=("config","scenario","partition_sha256","selection_sha256","initial_state_sha256",
          "code_sha256","versions","validation_tasks","evaluation_recorder","official_test_used","evaluation_data")
    for key in keys:
        if left.get(key)!=right.get(key):
            raise ValueError(f"Cross-method pair mismatch: {key}")


def extract_run(result,reference=None):
    """Return metrics and curves, using observed task weights for both paths."""
    ident=identity(result)
    scenario=longer_scenarios()[ident["scenario"]]
    overall=rounds_only(task_series(result))
    metrics={"overall_mean_5_24_pp":100*mean_window(overall,5,24),
             "overall_tail_21_24_pp":100*mean_window(overall,21,24),
             "overall_round24_pp":100*overall[24]}
    series_by_group={"all_clients":overall}
    honest_changes=[e for e in scenario.events if e.kind=="input_change" and e.intent=="honest"]
    attacks=[e for e in scenario.events if e.intent=="malicious"]
    changed={i for event in honest_changes for i in event.clients}
    adversarial={i for event in attacks for i in event.clients}
    unchanged=sorted(set(range(10))-changed-adversarial)
    if unchanged:
        series_by_group["unchanged_honest"]=rounds_only(task_series(result,client_ids=unchanged))
        metrics["unchanged_honest_mean_5_24_pp"]=100*mean_window(series_by_group["unchanged_honest"],5,24)
    if changed:
        affected=rounds_only(task_series(result,client_ids=sorted(changed)))
        series_by_group["affected_honest"]=affected
        end=10 if ident["scenario"]=="honest_change_stops" else 24
        metrics["affected_honest_event_pp"]=100*mean_window(affected,5,end)
        if ident["scenario"]=="honest_change_stops":
            metrics["affected_honest_return_11_24_pp"]=100*mean_window(affected,11,24)
    curves=[{**ident,"group":group,"round":r,"accuracy_pp":100*v}
            for group,series in series_by_group.items() for r,v in series.items()]
    model_episodes=[]
    if reference is not None:
        checked_pair(result,reference)
        # A shifted-task run and its clean control must be scored on the SAME task.
        ref=rounds_only(task_series(reference,scenario_name=ident["scenario"]))
        if attacks:
            windows=attacks[0].windows
            active=paired_damage(ref,overall,windows)
            metrics.update({"attack_active_signed_damage_pp":active["mean_signed_damage_pp"],
                            "attack_active_positive_area_pp_rounds":active["positive_damage_area_pp_rounds"],
                            "attack_tail_damage_21_24_pp":100*(mean_window(ref,21,24)-mean_window(overall,21,24))})
            if ident["scenario"]=="recurring_attack":
                metrics["second_attack_signed_damage_pp"]=paired_damage(ref,overall,((15,18),))["mean_signed_damage_pp"]
        for event in scenario.events:
            clients=list(event.clients)
            observed_series=rounds_only(task_series(result,client_ids=clients))
            reference_series=rounds_only(task_series(reference,scenario_name=ident["scenario"],client_ids=clients))
            for index,(start,end) in enumerate(event.windows):
                next_event=event.windows[index+1][0] if index+1<len(event.windows) else None
                model_episodes.append({**ident,"event_kind":event.kind,"intent":event.intent,"clients":clients,
                                       "episode":index+1,**rename_episode_flag(classify_model_episode(
                                           reference_series,observed_series,start,end,next_event))})
    return [{**ident,"metric":name,"value":value} for name,value in metrics.items()],curves,model_episodes


def contribution_records(result,reference):
    checked_pair(result,reference)
    ident=identity(result)
    scenario=longer_scenarios()[ident["scenario"]]
    observed=json.loads((ROOT/result["run_path"]/"aggregation_diagnostics.json").read_text())
    control=json.loads((ROOT/reference["run_path"]/"aggregation_diagnostics.json").read_text())
    obs={i:{row["round"]:row["values"]["client_metrics"][str(i)]["applied_contribution_l2_norm"] for row in observed} for i in range(10)}
    ref={i:{row["round"]:row["values"]["client_metrics"][str(i)]["applied_contribution_l2_norm"] for row in control} for i in range(10)}
    curves,episodes,reductions=[],[],[]
    changed={i for event in scenario.events if event.intent=="honest" for i in event.clients}
    attackers={i for event in scenario.events if event.intent=="malicious" for i in event.clients}
    for client in range(10):
        group="attacking" if client in attackers else "affected_honest" if client in changed else "unchanged_honest"
        ratios=contribution_norm_ratios(ref[client],obs[client])
        for r in range(1,25):
            item=observed[r-1]["values"]["client_metrics"][str(client)]
            curves.append({**ident,"client":client,"group":group,"round":r,"matched_norm_ratio":ratios[r],
                           "applied_norm":obs[client][r],"reference_norm":ref[client][r],
                           "coefficient":item["applied_delta_coefficient"],"evidence":item.get("evidence"),
                           "trust":item.get("trust"),"state":item.get("label"),
                           "applied_multiplier":item.get("applied_multiplier"),"allowance":item.get("allowance")})
        # All clients share the intervention windows for affected/unchanged comparisons.
        windows=sorted({(a,b) for event in scenario.events for a,b in event.windows})
        for index,(start,end) in enumerate(windows):
            later=[a for a,b in windows if a>end]
            next_event=min(later) if later else None
            for threshold in (.4,.5,.6):
                reductions.append({**ident,"client":client,"group":group,"event_start":start,"event_end":end,
                                   "maximum_ratio":threshold,**contribution_reduction_event(ref[client],obs[client],start,end,maximum_ratio=threshold)})
            for threshold in (.7,.8,.9):
                episodes.append({**ident,"client":client,"group":group,"minimum_ratio":threshold,
                                 **rename_episode_flag(classify_contribution_episode(ref[client],obs[client],start,end,next_event,minimum_ratio=threshold))})
    return curves,episodes,reductions


def cost_record(result):
    path=ROOT/result["run_path"]
    history=pd.read_csv(path/"history.csv")
    diagnostics=json.loads((path/"aggregation_diagnostics.json").read_text())
    base=[r["values"]["contribution_measurements"]["defence_aggregation_seconds"] for r in diagnostics]
    recorder=[r["values"]["contribution_measurements"]["metric_recording_seconds"] for r in diagnostics]
    return {**identity(result),"worker_wall_seconds":result["worker_wall_seconds"],
            "process_lifetime_peak_rss_mib":result["process_lifetime_peak_rss_bytes"]/2**20,
            "median_round_seconds":float(history.round_seconds.median()),
            "median_training_seconds":float(history.training_seconds.median()),
            "median_evaluation_seconds":float(history.evaluation_seconds.median()),
            "median_defence_seconds":float(np.median(base)),"median_recorder_seconds":float(np.median(recorder)),
            "reference_forward_examples":sum(r["values"].get("reference_forward_examples",0) for r in diagnostics),
            "note":"Concurrent fresh-process cost; RSS includes interpreter, data and framework; not isolated algorithm allocation."}


def reduction_seed_rates(reductions):
    rows=[]
    if not reductions: return rows
    keys=["phase","regime","seed","method","scenario","group","event_start","event_end","maximum_ratio"]
    for values,group in pd.DataFrame(reductions).groupby(keys):
        denominator=int(group.defined_client_rounds.sum())
        numerator=int(group.reduced_client_rounds.sum())
        rows.append({**group_fields(keys,values),"reduced_client_rounds":numerator,"defined_client_rounds":denominator,
                     "total_client_rounds":int(len(group)*(values[7]-values[6]+1)),
                     "reduction_fraction":numerator/denominator if denominator else None,
                     "meaning":"Descriptive matched contribution-norm reduction; this is not a malicious-intent classifier."})
    return rows


def episode_status_counts(episodes,group_keys):
    if not episodes: return []
    table=pd.DataFrame(episodes)
    rows=[]
    for values,group in table.groupby(group_keys,dropna=False):
        rows.append({**group_fields(group_keys,values if isinstance(values,tuple) else (values,)),
                     "episodes":len(group),"counts":{str(k):int(v) for k,v in group.status.value_counts().items()},
                     "note":"All statuses retained. Observed restoration delays apply only to post_event_restored episodes, not censored or already-within-criterion episodes."})
    return rows


def add_honest_composite(metrics):
    table=pd.DataFrame(metrics)
    if table.empty: return metrics
    chosen=table[(table.metric=="affected_honest_event_pp") & table.scenario.isin(HONEST_SCENARIOS)]
    for keys,group in chosen.groupby(["phase","regime","seed","method"]):
        if set(group.scenario)==set(HONEST_SCENARIOS) and len(group)==4:
            metrics.append({**group_fields(["phase","regime","seed","method"],keys),"scenario":"honest_composite",
                            "job_id":"composite_of_four_scenarios","metric":"honest_composite_pp","value":float(group.value.mean())})
    return metrics


def summarise_metrics(metrics,expected_n=5):
    rows=[]
    for keys,group in pd.DataFrame(metrics).groupby(["phase","regime","method","scenario","metric"]):
        if group.seed.duplicated().any(): raise ValueError("Duplicate seed in summary.")
        rows.append({**dict(zip(["phase","regime","method","scenario","metric"],keys)),**interval(group.value,expected_n)})
    return rows


def paired_contrasts(metrics,index,expected_n=5):
    table=pd.DataFrame(metrics)
    rows=[]
    for regime in PARTITIONS:
        subset=table[table.regime==regime]
        full=subset[subset.method=="FedReTrust"]
        for comparator in sorted(set(subset.method)-{"FedReTrust"}):
            comparison=subset[subset.method==comparator]
            joined=full.merge(comparison,on=["regime","seed","scenario","metric"],suffixes=("_full","_other"),validate="one_to_one")
            for (scenario,metric),group in joined.groupby(["scenario","metric"]):
                for seed in group.seed:
                    if scenario=="honest_composite":
                        for name in HONEST_SCENARIOS:
                            cross_method_pair(index[(regime,seed,"FedReTrust",name)],index[(regime,seed,comparator,name)])
                    else:
                        cross_method_pair(index[(regime,seed,"FedReTrust",scenario)],index[(regime,seed,comparator,scenario)])
                delta=group.value_full-group.value_other
                rows.append({"regime":regime,"contrast":"FedReTrust minus "+comparator,"comparator":comparator,
                             "scenario":scenario,"metric":metric,"seeds":group.seed.tolist(),
                             "paired_differences":delta.tolist(),**interval(delta,expected_n)})
    return rows


def assess_primary(contrasts,coverage_complete):
    lookup={(r["regime"],r["scenario"],r["metric"]):r for r in contrasts if r["comparator"]=="Viceroy"}
    rows=[]
    for regime in PARTITIONS:
        tests=[("honest_composite","honest_composite_pp","gain",1.)]
        tests += [("clean","overall_mean_5_24_pp","lower_guard",-1.)]
        tests += [(name,"affected_honest_event_pp","lower_guard",-1.) for name in HONEST_SCENARIOS]
        tests += [("honest_change_stops","affected_honest_return_11_24_pp","lower_guard",-1.)]
        tests += [(name,"attack_active_signed_damage_pp","upper_guard",1.) for name in ATTACK_SCENARIOS]
        for scenario,metric,kind,margin in tests:
            record=lookup.get((regime,scenario,metric))
            passed=False
            if record and record["complete_seed_coverage"]:
                if kind=="gain": passed=record["mean"]>=margin and record["ci_low"]>0
                elif kind=="lower_guard": passed=record["ci_low"]>=margin
                else: passed=record["ci_high"]<=margin
                point_meets=record["mean"]>=margin if kind!="upper_guard" else record["mean"]<=margin
                status="supported_by_descriptive_interval" if passed else "interval_inconclusive" if point_meets else "point_estimate_does_not_meet_margin"
            else: status="incomplete_seed_coverage"
            rows.append({"regime":regime,"scenario":scenario,"metric":metric,"criterion":kind,"margin_pp":margin,
                         "passed":bool(passed),"status":status,"estimate":record})
    overall=coverage_complete and all(row["passed"] for row in rows)
    return {"overall_predeclared_criterion_met":bool(overall),"complete_all_planned_runs":coverage_complete,"criteria":rows,
            "interpretation":"Descriptive evidence conditional on one dataset, fixed partitions/cohorts, five seeds and this bounded workload. Passing does not prove universal superiority or malicious intent identification."}


def make_figures(curves,contrasts,costs,out):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/fedretrust_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(curves)
    paths=[]
    names=list(longer_scenarios())[:10]
    for regime in PARTITIONS:
        fig,axes=plt.subplots(2,5,figsize=(20,8),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,names):
            data=df[(df.regime==regime)&(df.scenario==name)&(df.group=="all_clients")&df.method.isin(METHODS)]
            for method in METHODS:
                group=data[data.method==method].groupby("round").accuracy_pp
                if group.ngroups:
                    mean,sd=group.mean(),group.std()
                    ax.plot(mean.index,mean,label=method,linewidth=1.5)
                    ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.1)
            ax.set_title(name.replace("_"," "),fontsize=10); ax.grid(alpha=.2); ax.set_ylim(0,100)
            ax.set_xlabel("Round")
        axes[0,0].set_ylabel("Legitimate-task accuracy (%)"); axes[1,0].set_ylabel("Legitimate-task accuracy (%)")
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"{regime}: paired-seed means; shaded bands are ±1 sample SD")
        fig.tight_layout()
        path=out/f"main_curves_{regime}.png"; fig.savefig(path,dpi=160); plt.close(fig); paths.append(str(path))
    # Mechanism contrast uses the identical second attack window for both variants.
    relevant=[r for r in contrasts if r["comparator"]=="FedReTrust:no_repeat_memory" and r["metric"]=="second_attack_signed_damage_pp"]
    if relevant:
        fig,ax=plt.subplots(figsize=(7,4))
        for i,row in enumerate(relevant):
            ax.scatter([i]*len(row["paired_differences"]),row["paired_differences"],alpha=.55)
            if row["ci_low"] is not None:
                ax.errorbar(i,row["mean"],yerr=[[row["mean"]-row["ci_low"]],[row["ci_high"]-row["mean"]]],fmt="ks",capsize=5)
        ax.set_xticks(range(len(relevant)),[r["regime"] for r in relevant]); ax.axhline(0,color="grey",linestyle="--")
        ax.set_ylabel("Excess second-episode damage (pp)\nFull minus no-repeat-memory; lower is better")
        ax.set_title("Same rounds 15–18; paired seeds and descriptive 95% intervals")
        fig.tight_layout(); path=out/"recurrence_component.png";fig.savefig(path,dpi=160);plt.close(fig);paths.append(str(path))
    gains=[r for r in contrasts if r["comparator"] in METHODS and r["metric"]=="honest_composite_pp"]
    if gains:
        fig,ax=plt.subplots(figsize=(9,5))
        for i,row in enumerate(gains):
            ax.scatter(row["paired_differences"],[i]*len(row["paired_differences"]),alpha=.45)
            if row["ci_low"] is not None:
                ax.errorbar(row["mean"],i,xerr=[[row["mean"]-row["ci_low"]],[row["ci_high"]-row["mean"]]],fmt="ks",capsize=4)
        ax.set_yticks(range(len(gains)),[f"{r['regime']} / {r['comparator']}" for r in gains])
        ax.axvline(0,color="grey",linestyle="--");ax.axvline(1,color="green",linestyle=":")
        ax.set_xlabel("Affected-honest composite gain (pp): FedReTrust minus comparator")
        ax.set_title("Paired seeds with descriptive 95% intervals; dotted line = proposed 1 pp margin")
        fig.tight_layout();path=out/"honest_composite_contrasts.png";fig.savefig(path,dpi=160);plt.close(fig);paths.append(str(path))
    table=pd.DataFrame(costs)
    if not table.empty:
        main=table[table.method.isin(METHODS)]
        data=main.groupby("method")[["median_defence_seconds","median_training_seconds","median_evaluation_seconds"]].median().reindex(METHODS)
        fig,ax=plt.subplots(figsize=(8,4));data.plot.bar(stacked=True,ax=ax)
        ax.set_ylabel("Seconds per round, median across runs");ax.set_title("Observed concurrent CPU costs; instrumentation excluded from bars")
        ax.tick_params(axis="x",rotation=0);ax.legend(["Defence","Local training","Evaluation"],fontsize=8)
        fig.tight_layout();path=out/"execution_costs.png";fig.savefig(path,dpi=160);plt.close(fig);paths.append(str(path))
    return paths


def analyse(study=DEFAULT_STUDY):
    study=Path(study)
    freeze=verify_freeze(study/"frozen_protocol.json")
    results=[result_for(study,j) for j in freeze["authorized_final_jobs"]]
    if any(result is None for result in results):
        raise ValueError("Final, control or sensitivity jobs are still pending; do not label analysis complete.")
    out=study/"analysis"; out.mkdir(parents=True,exist_ok=True)
    index={run_key(r["job"]):r for r in results if r["status"]=="complete"}
    coverage=[{**identity(r),"status":r["status"],"error":r.get("error")} for r in results]
    metrics,curves,model_episodes,costs,contribution_episodes,reductions=[],[],[],[],[],[]
    contribution_path=out/"contribution_curves.csv"
    first=True
    for result in results:
        if result["status"]!="complete": continue
        for name,digest in result["artifact_sha256"].items():
            if file_sha256(ROOT/result["run_path"]/name)!=digest:
                raise ValueError(f"Saved artifact changed before analysis: {result['job_id']}/{name}")
        job=result["job"]
        reference=index.get(run_key(job,matched_reference_name(job["scenario"]))) if job["scenario"]!="clean" else None
        values,trajectory,episodes=extract_run(result,reference)
        metrics+=values;curves+=trajectory;model_episodes+=episodes;costs.append(cost_record(result))
        if reference is not None:
            traces,episodes,reduction=contribution_records(result,reference)
            pd.DataFrame(traces).to_csv(contribution_path,index=False,mode="w" if first else "a",header=first)
            first=False; contribution_episodes+=episodes;reductions+=reduction
    add_honest_composite(metrics)
    summaries=summarise_metrics(metrics)
    contrasts=paired_contrasts(metrics,index)
    complete=all(r["status"]=="complete" for r in results)
    primary=assess_primary(contrasts,complete)
    tables={"coverage":coverage,"seed_metrics":metrics,"seed_summaries":summaries,"paired_contrasts":contrasts,
            "accuracy_curves":curves,"model_episodes":model_episodes,"contribution_episodes":contribution_episodes,
            "contribution_reduction_sensitivity":reductions,"costs":costs,
            "contribution_reduction_seed_rates":reduction_seed_rates(reductions),
            "model_episode_status_counts":episode_status_counts(model_episodes,["regime","method","scenario","event_kind","intent","episode"]),
            "contribution_episode_status_counts":episode_status_counts(contribution_episodes,["regime","method","scenario","group","minimum_ratio","event_start_round","event_end_round"])}
    for name,records in tables.items():
        save(out/f"{name}.json",records)
        pd.json_normalize(records).to_csv(out/f"{name}.csv",index=False)
    save(out/"primary_assessment.json",primary)
    figures=make_figures(curves,contrasts,costs,out/"figures")
    files=sorted(p for p in out.rglob("*") if p.is_file() and p.name!="analysis_manifest.json")
    manifest={"status":"complete","at":now(),"freeze_sha256":file_sha256(study/"frozen_protocol.json"),
              "analysis_source_sha256":{name:file_sha256(ROOT/"code"/name) for name in ("research_analysis.py","research_study.py")},
              "planned_runs":len(results),"completed_runs":sum(r["status"]=="complete" for r in results),
              "numerical_failures":sum(r["status"]!="complete" for r in results),
              "overall_predeclared_criterion_met":primary["overall_predeclared_criterion_met"],
              "figures":figures,"artifacts":{str(p.relative_to(out)):file_sha256(p) for p in files},
              "limitations":[freeze["fixed_partition_limits"],freeze["horizon_limits"],freeze["workload_scope"],
                             "Trusted clean root is a shared extra-information assumption for FLTrust and FedReTrust.",
                             "No optimal-baseline claim: only predeclared server-rate calibration with fixed defence constants.",
                             "Contribution norms are delivered-update magnitudes, not causal influence or intent classification.",
                             "First qualifying restoration streak does not establish permanent restoration.",
                             "Missing numerical-failure seeds are shown explicitly; no complete-evidence claim uses a reduced subset."]}
    save(out/"analysis_manifest.json",manifest)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--study",type=Path,default=DEFAULT_STUDY)
    args=parser.parse_args();print(json.dumps(analyse(args.study),indent=2))
