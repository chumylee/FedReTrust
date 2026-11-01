"""Build and execute experiment notebooks from the saved study artifacts.

InteractiveShell avoids TCP kernel sockets in the restricted execution sandbox.
The saved notebooks remain ordinary Jupyter notebooks with the project kernel.
"""
from pathlib import Path
import argparse
import os
import nbformat as nbf

from research_runs import ROOT

COMMON = """from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image
CODE = Path.cwd() if (Path.cwd()/'research_study.py').exists() else Path.cwd()/'Experiment'/'code'
sys.path.insert(0,str(CODE.resolve()))
from research_study import ROOT, DEFAULT_STUDY, read_plan, result_for, task_series, run_stage
from research_analysis import interval
STUDY = DEFAULT_STUDY
PLAN = read_plan(STUDY)
def read_optional(name):
    path = STUDY/name
    return json.loads(path.read_text()) if path.exists() else None
def progress(stage):
    value = read_optional('progress_'+stage+'.json')
    return {key:value[key] for key in ('stage','status','updated_at','total','completed','numerical_failures')} if value else {'stage':stage,'status':'not started'}
"""


def contents(number):
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    if number==13:
        return [
            md('# 13 — Development calibration and protocol freeze\n\n**Roadmap Step 7.** Run the same predeclared development scenarios for all five methods. Select one server rate per method and data regime, then freeze the complete final manifest before opening the official test set. Defence parameters stay fixed. This is limited server-rate calibration, not exhaustive algorithm tuning.'),
            code(COMMON),
            md('## Shared workload and candidate budget\n\nThe 512 training images per client are fixed within each run and selected afresh by seed. Full saved client class frequencies define the evaluation task. FLTrust and FedReTrust share the same reserved 300-image root reference and eight root optimizer steps. FedReTrust’s additional loss probes are timed.'),
            code("display(pd.Series(PLAN['workload'],name='Declared workload').to_frame())\ndisplay(pd.Series(PLAN['job_counts'],name='Number of runs').to_frame())\nprint('Candidate relative rates:',PLAN['candidate_relative_rates'])\nprint(PLAN['selection_rule'])\nprint(PLAN['candidate_rationale'])"),
            md('## Common feasibility panel\n\nAll five methods are checked in both regimes under clean training, a temporary honest input change and a recurring attack. Passing requires operability and basic clean FedAvg learning; FedReTrust does not need to win. The 24-round horizon can censor recovery.'),
            code("display(pd.DataFrame([progress('feasibility'),progress('development')]))\nreview=read_optional('feasibility_review.json')\nif review:\n    print('Feasibility gate passed:',review['passed'])\n    display(pd.DataFrame(review['results'])[['regime','method','scenario','status','accuracy_round24','late_gain_pp','median_round_seconds','peak_rss_bytes']])\nelse:\n    print('Feasibility panel still in progress; no protocol freeze yet.')"),
            md('## Selected settings\n\nThe objective averages legitimate-task accuracy equally across rounds 5–24, the eight primary scenarios and three development seeds. A numerical failure makes the entire candidate ineligible. Boundary winners are disclosed; the grid is not extended after outcomes.'),
            code("selection=read_optional('development_selection.json')\nif selection:\n    display(pd.DataFrame(selection['candidates'])[['regime','method','relative_rate','eligible','mean_accuracy','seed_scores']])\n    display(pd.DataFrame([row for methods in selection['selected'].values() for row in methods.values()])[['regime','method','relative_rate','mean_accuracy','boundary_winner']])\nelse:\n    print('Development selection is not complete.')\nfreeze=read_optional('frozen_protocol.json')\nprint('Final protocol:',freeze['status'] if freeze else 'not frozen; official test remains reserved')\nif freeze: print('Frozen at:',freeze['frozen_at'],'; final manifest jobs:',len(freeze['authorized_final_jobs']))"),
            md('## Reproduce an execution stage\n\nThe coordinator used for these runs is callable below. Leave `RUN_STAGE = None` while another coordinator is active. To resume manually, choose `feasibility` or `development`; the worker verifies completed artifacts and runs remaining jobs. Review and freeze are explicit commands documented in the README.'),
            code("RUN_STAGE = None\nif RUN_STAGE is not None:\n    run_stage(RUN_STAGE,STUDY)\nelse:\n    print('Viewing saved results; no duplicate coordinator started.')"),
            md('**Next — Step 8:** execute the frozen five-method final comparison with five independent paired seeds in each regime. Sol Extra High is the recommended execution model; Astra Extra High can review fairness and the interpretation of the development selection.')]
    if number==14:
        return [
            md('# 14 — Frozen five-method comparison\n\n**Roadmap Step 8.** Compare FedReTrust with FedAvg, FLTrust, RFA and Viceroy using the same selected training data, initialization, scenarios and independent final seeds. All 10,000 official test images are evaluated after protocol freezing. No final outcome changes the selected settings.'),
            code(COMMON),
            code("freeze=read_optional('frozen_protocol.json')\nassert freeze is not None, 'Complete development and freeze the protocol first.'\ndisplay(pd.DataFrame([progress('final')]))\nprint('Final seeds:',freeze['final_seeds'])\nprint(freeze['workload_scope'])\nprint(freeze['horizon_limits'])"),
            md('## Completed-run coverage and final accuracy\n\nEvery planned method/scenario/regime has five seeds. The table uses population-standardized legitimate-task accuracy averaged over rounds 5–24. Failed seeds are visible and do not silently disappear into a favourable mean. Intervals are descriptive and conditional on the fixed partitions.'),
            code("coverage=[]; values=[]\nfor request in freeze['authorized_final_jobs']:\n    if request['phase'] != 'final': continue\n    result=result_for(STUDY,request)\n    status=result['status'] if result else 'pending'\n    coverage.append({k:request[k] for k in ('method','regime','scenario','seed')}|{'status':status})\n    if status=='complete':\n        metadata=json.loads((ROOT/result['run_path']/'run.json').read_text())\n        assert metadata['official_test_used']\n        series=task_series(result)\n        values.append({k:request[k] for k in ('method','regime','scenario','seed')}|{'accuracy_pp':100*sum(series[r] for r in range(5,25))/20})\ndisplay(pd.DataFrame(coverage).groupby(['regime','method','status']).size().rename('runs').reset_index())\nsummary=[]\nif values:\n    for keys,group in pd.DataFrame(values).groupby(['regime','method','scenario']):\n        summary.append(dict(zip(['regime','method','scenario'],keys))|interval(group.accuracy_pp))\n    display(pd.DataFrame(summary))"),
            md('## Accuracy trajectories\n\nThese figures are produced by the completed offline analysis. They show seed means with ±1 sample SD, rather than a selected best run.'),
            code("for regime in PLAN['regimes']:\n    path=STUDY/'analysis'/'figures'/('main_curves_'+regime+'.png')\n    if path.exists(): display(Image(filename=str(path)))\n    else: print(regime+': final analysis figure is not available yet.')"),
            md('**Next — Step 9:** run the predeclared component controls and parameter sensitivity. Sol Extra High executes; Astra Extra High reviews whether the controls test the intended mechanisms.')]
    if number==15:
        return [
            md('# 15 — Component controls and sensitivity\n\n**Roadmap Step 9.** The seven controls each run their own trajectory with the selected full-method settings. Clean and mixed-drift reference runs are included wherever needed. The three parameter checks change one constant at a time and inherit the selected server rate; they are not a second tuning stage.'),
            code(COMMON),
            code("display(pd.DataFrame([progress('ablation'),progress('sensitivity')]))\ndisplay(pd.DataFrame([{'variant':v,'scenarios':', '.join(s)} for v,s in PLAN['targeted_controls'].items()]))\ndisplay(pd.DataFrame(PLAN['parameter_sensitivity']))"),
            md('## Paired changes from the full method\n\nAccuracy differences use full FedReTrust minus the control: positive values favour the full method. Damage differences use the same subtraction: negative values favour the full method. All contrasts pair the same seed and scenario. The mass-only control matches its own current clipped coefficient mass, not the full method’s model trajectory or update norm.'),
            code("path=STUDY/'analysis'/'paired_contrasts.json'\nif path.exists():\n    contrasts=pd.DataFrame(json.loads(path.read_text()))\n    selected=contrasts[contrasts.comparator.str.startswith('FedReTrust:')]\n    display(selected[['regime','comparator','scenario','metric','n','mean','ci_low','ci_high']])\nelse:\n    print('Offline analysis is pending; no mechanism conclusion yet.')"),
            md('## Recurrence and operational-threshold checks\n\nRepeat-memory evidence compares full vs no-repeat-memory during the same second attack window, rounds 15–18. The first and second attacks occur at different learning stages and are not a causal before/after control. Contribution-norm thresholds describe delivered-update magnitudes; they do not identify malicious intent.'),
            code("figure=STUDY/'analysis'/'figures'/'recurrence_component.png'\nif figure.exists(): display(Image(filename=str(figure)))\npath=STUDY/'analysis'/'contribution_reduction_seed_rates.json'\nif path.exists():\n    rates=pd.DataFrame(json.loads(path.read_text()))\n    display(rates[(rates.method=='FedReTrust') & rates.group.isin(['affected_honest','attacking'])].groupby(['regime','scenario','group','maximum_ratio']).reduction_fraction.agg(['count','mean','std']).reset_index())\nelse:\n    print('Saved-trace sensitivity is pending.')"),
            md('**Next — Step 10:** assess the complete evidence, paired-seed uncertainty and limitations. Sol Extra High produces the statistics; Astra Extra High is recommended for the final scientific interpretation.')]
    if number==16:
        return [
            md('# 16 — Results, claims and limitations\n\n**Roadmap Step 10.** Assess the predeclared question using all planned runs. The primary contrast is FedReTrust versus Viceroy, with all four baseline comparisons shown. Implementation correctness and empirical usefulness are separate questions. A criterion failure is retained as a research result.'),
            code(COMMON),
            code("manifest=read_optional('analysis/analysis_manifest.json')\nassert manifest and manifest['status']=='complete', 'Complete all final/control/sensitivity runs and offline analysis first.'\nassessment=read_optional('analysis/primary_assessment.json')\nprint('Completed runs:',manifest['completed_runs'],'/',manifest['planned_runs'])\nprint('Numerical failures:',manifest['numerical_failures'])\nprint('Overall predeclared criterion met:',assessment['overall_predeclared_criterion_met'])\nprint(assessment['interpretation'])"),
            md('## Primary evidence and guard conditions\n\nThe affected-honest composite must gain at least 1 percentage point on average with a descriptive interval above zero. Clean accuracy, each honest component and return accuracy have separate 1-point loss guards. Each attack scenario has its own 1-point excess-damage guard. All conditions must hold in both regimes; a favourable pooled average cannot hide a failed guard.'),
            code("rows=[]\nfor item in assessment['criteria']:\n    estimate=item['estimate'] or {}\n    rows.append({k:item[k] for k in ('regime','scenario','metric','criterion','margin_pp','status')}|{k:estimate.get(k) for k in ('n','mean','ci_low','ci_high')})\ndisplay(pd.DataFrame(rows))\nfigure=STUDY/'analysis'/'figures'/'honest_composite_contrasts.png'\nif figure.exists(): display(Image(filename=str(figure)))"),
            md('## Recovery, contribution and computation\n\nEvery episode status is retained, including no observed impairment, restoration within the event, delayed impairment, undefined comparisons and right censoring. A first three-round restoration streak does not establish permanent recovery. Runtime and fresh-worker peak RSS include the common framework, data and concurrent resource contention.'),
            code("episodes=read_optional('analysis/model_episode_status_counts.json')\ndisplay(pd.json_normalize(episodes))\ncosts=pd.DataFrame(read_optional('analysis/costs.json'))\ndisplay(costs[costs.method.isin(PLAN['methods'])].groupby(['regime','method'])[['median_defence_seconds','median_training_seconds','median_evaluation_seconds','process_lifetime_peak_rss_mib']].median())\nfigure=STUDY/'analysis'/'figures'/'execution_costs.png'\nif figure.exists(): display(Image(filename=str(figure)))"),
            md('## What the evidence can support\n\nThe results are conditional on MNIST, the fixed IID/non-IID partitions and cohorts, the 512-image client workload, the clean root reference assumption and the 24-round horizon. Five-seed intervals are descriptive and are not adjusted for all the comparisons. These experiments do not establish an accurate classifier of hidden intent or universal superiority.'),
            code("for limitation in manifest['limitations']: print('•',limitation)\nprint('Analysis record:',STUDY/'analysis'/'analysis_manifest.json')\nprint('Frozen protocol:',STUDY/'frozen_protocol.json')\nif assessment['overall_predeclared_criterion_met']:\n    print('The declared overall criterion is supported within this study. Discuss scope and replication with the supervisor.')\nelse:\n    print('The declared overall criterion is not established. Bring the specific failed or inconclusive guards and component results to the supervisor; do not retune from these test outcomes.')"),
            md('**Supervisor checkpoint reached after Step 10 once these results are complete.** Bring the mathematical draft, implementation checks, frozen protocol, five-method results, controls and limitations. **Step 11** is the final report, presentation and reproducibility package; Sol Extra High can assemble it and Astra Extra High can review scientific claims.')]
    raise ValueError('Choose notebook 13, 14, 15 or 16.')


NAMES={13:'13_development_and_freeze',14:'14_final_comparison',15:'15_ablations_and_sensitivity',16:'16_results_analysis'}


def build(number,execute=True,study=None):
    path=ROOT/'code'/(NAMES[number]+'.ipynb')
    notebook=nbf.v4.new_notebook(cells=contents(number))
    if study is not None:
        for cell in notebook.cells:
            if cell.cell_type=='code' and cell.source==COMMON:
                cell.source=cell.source.replace('STUDY = DEFAULT_STUDY',f'STUDY = Path({str(Path(study).resolve())!r})')
    notebook.metadata.kernelspec={'display_name':'FedReTrust Experiment (Python 3.14)','language':'python','name':'fedretrust-experiment'}
    nbf.write(notebook,path)
    if execute:
        os.environ.setdefault('IPYTHONDIR','/tmp/fedretrust_ipython')
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        shell=InteractiveShell.instance()
        previous=Path.cwd();os.chdir(path.parent)
        try:
            count=0
            for cell in notebook.cells:
                if cell.cell_type!='code': continue
                count+=1
                with capture_output() as capture:
                    result=shell.run_cell(cell.source,store_history=False)
                cell.execution_count=count;cell.outputs=[]
                if capture.stdout:cell.outputs.append(nbf.v4.new_output('stream',name='stdout',text=capture.stdout))
                if capture.stderr:cell.outputs.append(nbf.v4.new_output('stream',name='stderr',text=capture.stderr))
                for item in capture.outputs:cell.outputs.append(nbf.v4.new_output('display_data',data=item.data,metadata=item.metadata))
                nbf.write(notebook,path)
                if result.error_before_exec or result.error_in_exec:
                    raise RuntimeError(f'Notebook {number}, code cell {count}: {result.error_before_exec or result.error_in_exec}')
        finally:os.chdir(previous)
        notebook.metadata['execution_note']='All code cells executed with IPython InteractiveShell in the project virtual environment; TCP kernel sockets are unavailable in this sandbox.'
        nbf.write(notebook,path)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('number',type=int,choices=NAMES);parser.add_argument('--no-execute',action='store_true')
    args=parser.parse_args();print(build(args.number,not args.no_execute))
