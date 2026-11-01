"""Execute unchanged v0.6 on tiny synthetic vectors; not a performance experiment."""
import argparse
import json
import sys
from common import ROOT, new_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='walkthrough', help='Unique output label.')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'Experiment/code'))
    import torch
    from experiment_core import ClientSubmission
    from fedretrust_v06 import FedReTrustV06
    from v06_recording import InnovationContributionRecorder

    torch.set_num_threads(1)
    # Illustrations of the implemented mechanism, not simulated MNIST training.
    cases = [
        ('History filtering', [
            ('Aligned large updates can be clipped at full trust', [[1., 0.]] * 7 + [[100., 0.]] * 3),
            ('A raw reversal can remain aligned after history filtering', [[1.1, 0.]] * 7 + [[-10., 0.]] * 3),
        ]),
        ('Directional trust; fresh method state', [
            ('Opposing submissions at cold start', [[1., 0.]] * 7 + [[-1., 0.]] * 3),
            ('Aligned submissions return; inspect zero-median bypass too', [[1., 0.]] * 10),
            ('Another aligned participation', [[1., 0.]] * 10),
        ]),
    ]
    out = new_output('demo', args.name)
    trace = []
    print('MECHANISM DEMO ONLY — synthetic vectors; no MNIST accuracy, study run or attack-recovery finding.\n')
    for case, stages in cases:
        method = InnovationContributionRecorder(FedReTrustV06())
        broadcast = {'x': torch.zeros(2, dtype=torch.float64)}
        print(case)
        for round_number, (label, points) in enumerate(stages, 1):
            submissions = [ClientSubmission(i, 512, {'x': broadcast['x'] + torch.tensor(point, dtype=torch.float64)})
                           for i, point in enumerate(points)]
            result = method.aggregate(submissions, broadcast, round_number)
            delta = result.state['x'] - broadcast['x']
            row = {'case': case, 'round': round_number, 'illustration': label, 'raw_updates': points,
                   'global_update': delta.tolist(), 'diagnostics': result.diagnostics}
            trace.append(row)
            client = result.diagnostics['client_metrics']['7']
            print(f'Round {round_number}: {label}')
            print(f'  Global update: {delta.tolist()}')
            print(f"  Client 7: innovation factor={client['innovation_factor']:.4f}; "
                  f"trust={client['restored_weight']}; current coefficient={result.coefficients[7]:.4f}; "
                  f"history coefficient={client['historical_coefficient']:.4f}")
            print(f"  RFA complement mass: {result.diagnostics['robust_complement_mass']:.4f}; "
                  f"zero-median bypass: {result.diagnostics['zero_median_innovation_bypass']}\n")
            broadcast = result.state
    write_json(out / 'trace.json', {'scope': 'Synthetic mechanism illustration only', 'rounds': trace})
    print('Trace saved to', out.relative_to(ROOT) / 'trace.json')


if __name__ == '__main__':
    main()
