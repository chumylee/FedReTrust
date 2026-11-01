"""Check coverage, declared access, and failed-gate history in saved CSVs."""
import csv
import json
from collections import Counter
from common import TABLES


def read(name):
    with (TABLES / (name + '.csv')).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    runs, access, settings = read('run_index'), read('data_access_by_run'), read('run_settings')
    def identity(row):
        return tuple(row[k] for k in ('method', 'regime', 'block', 'scenario', 'job_id'))
    ids = {identity(row) for row in runs}
    if len(runs) != len(access) or len(runs) != len(settings) or len(ids) != 880:
        raise ValueError('Expected 880 distinct indexed runs and matching access/settings rows.')
    if ids != {identity(row) for row in access} or ids != {identity(row) for row in settings}:
        raise ValueError('Run/access/settings identities differ.')
    if any(row['official_test_used'] != 'False' or row['defence_uses_evaluation_data'] != 'False'
           for row in access):
        raise ValueError('Saved access record contradicts declared evaluation isolation.')
    candidate = [row for row in runs if row['method'] == 'FedReTrust-v06']
    actual = Counter((row['regime'], row['scenario']) for row in candidate)
    expected = Counter({(regime, scenario): 5 for regime in ('IID', 'non-IID')
                        for scenario in ('clean', 'abrupt_shot_noise', 'gradual_stripe', 'label_flip')})
    if actual != expected or len(candidate) != 40:
        raise ValueError('v0.6 run coverage differs from five blocks x two regimes x four scenarios.')
    if any(row['server_reference_examples'] != '0' for row in access if row['method'] == 'FedReTrust-v06'):
        raise ValueError('v0.6 reference-image access differs.')
    gates, failures = read('v06_followup_gates'), read('v06_failed_targets')
    def key(row):
        fields = tuple(row[k] for k in ('candidate', 'comparator', 'regime', 'scenario',
                                        'metric', 'criterion', 'assessment_unit'))
        block = row['gate_block']
        return (*fields, None if block in ('', 'NA') else int(float(block)))
    failed = [row for row in gates if row['point_pass'] == 'False']
    if len(gates) != 51 or len(failed) != 6 or len(failures) != 6 or {key(row) for row in failures} != {key(row) for row in failed}:
        raise ValueError('Follow-up gate/failure history differs.')
    print(json.dumps({'status': 'saved_evidence_consistent', 'distinct_runs': len(ids),
        'v06_runs': len(candidate), 'v06_cells': len(actual),
        'official_test_used_runs': 0, 'defence_evaluation_data_access_runs': 0,
        'v06_server_reference_images_per_run': 0, 'followup_passed': len(gates) - len(failed),
        'followup_failed': len(failed),
        'scope': 'Saved-record consistency; not a new experiment or proof of unknown external data access.'}, indent=2))


if __name__ == '__main__':
    main()
