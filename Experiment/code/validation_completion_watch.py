"""Finish the saved-artifact report when the already-running study completes.

This companion does not start, retry or tune experiments. A bounded wait and an
exclusive lock keep it safe to resume. The training coordinator remains the
sole owner of scientific execution and its status file.
"""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/validation_replication/20261030_v03_v1'


def record(status, **kwargs):
    path = STUDY / 'completion_watch.json'
    value = {'status': status, 'at': datetime.now(timezone.utc).isoformat(), **kwargs}
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    with (STUDY / 'completion_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        while time.monotonic() - start < 24 * 3600:
            status = json.loads((STUDY / 'status.json').read_text())
            if status['status'] == 'stopped_on_error':
                record('needs_attention', coordinator_status=status)
                raise RuntimeError('Scientific coordinator stopped; preserve its failure and investigate.')
            if status['status'] == 'complete':
                if status.get('completed_new_runs') != 800:
                    raise ValueError('Completion count must be 800.')
                record('building_report')
                with (STUDY / 'presentation_build.log').open('a') as log:
                    subprocess.run([sys.executable, str(ROOT / 'code/validation_presentation.py')],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                record('report_ready_for_visual_review', notebook='code/29_fedretrust_validation.ipynb')
                return
            progress_path = STUDY / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            record('waiting_for_study', completed=progress.get('completed', 0), total=800)
            time.sleep(30)
        record('needs_attention', reason='24-hour companion deadline reached; training was not terminated.')
        raise TimeoutError('Completion companion deadline reached.')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        # Never rewrite the scientific status or discard a failed job.
        record('needs_attention', error=repr(exc))
        raise
