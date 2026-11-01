"""Coverage, sensitivity isolation and calibration independence checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import flaura_study as study
from scenario_metrics import matched_reference_name

class FLAURAStudyTests(unittest.TestCase):
    def test_full_declared_coverage_and_unique_keys(self):
        jobs=study.jobs()
        self.assertEqual(len(jobs),68)
        self.assertEqual(len({study.key(j) for j in jobs}),68)
        self.assertEqual(sum(j['method']==study.MAIN for j in jobs),30)
        self.assertEqual(sum(j['method']==study.CONTROL for j in jobs),12)
        self.assertEqual(sum(j['flaura_stage']=='sensitivity' for j in jobs),24)
        self.assertEqual(sum(j['method']==study.CALIBRATION for j in jobs),2)
        self.assertTrue(all(j['phase']=='development' for j in jobs))

    def test_every_event_has_own_matched_reference(self):
        keys={study.key(j) for j in study.jobs()}
        for j in study.jobs():
            self.assertIn((j['method'],j['regime'],matched_reference_name(j['scenario']),j['seed']),keys)

    def test_one_at_a_time_settings_and_no_filter_control(self):
        main=study.SETTINGS[study.MAIN]
        for name,config in study.SETTINGS.items():
            if name!=study.MAIN:
                self.assertEqual(sum(config[k]!=main[k] for k in main),1)
        self.assertEqual(study.SETTINGS[study.CONTROL]['gamma'],0.)

    def test_calibration_is_disjoint_in_seed_and_does_not_read_accuracy(self):
        self.assertNotIn(study.CALIBRATION_SEED,study.SEEDS)
        with TemporaryDirectory() as temp:
            root=Path(temp);s=root/'study';s.mkdir()
            (s/'protocol.json').write_text(json.dumps({'calibration':{'rule':'median'}}))
            runs=[]
            for i in range(2):
                p=root/f'run{i}';p.mkdir()
                rows=[{'values':{'plr_center_distances':list(range(i*60+r*10,i*60+(r+1)*10))}} for r in range(6)]
                (p/'aggregation_diagnostics.json').write_text(json.dumps(rows))
                runs.append({'job':{'method':study.CALIBRATION},'job_id':str(i),'run_path':p.name,'artifact_sha256':{}})
            with patch.object(study,'ROOT',root),patch.object(study,'records',return_value=runs):
                freeze=study.freeze_scale(s)
                self.assertEqual(freeze['sigma_g'],59.5)
                self.assertEqual(study.freeze_scale(s),freeze)
                changed=json.loads((s/'scale_freeze.json').read_text());changed['sigma_g']=10
                (s/'scale_freeze.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):study.freeze_scale(s)

    def test_calibration_must_complete_both_regimes(self):
        with patch.object(study,'records',return_value=[]):
            with self.assertRaises(ValueError):study.freeze_scale(Path('/unused'))

if __name__=='__main__':unittest.main()
