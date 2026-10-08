import torch

from tartan.research_score.evaluation.metrics_navigation import checkpoint_selection_key
from tartan.research_score.scripts import train_transfer


def row(sr, spl, cr=0.1, progress=0.5, update=250):
    return {"navigation": {"sr": sr, "spl": spl, "cr": cr, "goal_progress": progress},
            "target_updates": update}


def test_checkpoint_selection_matches_protocol():
    assert checkpoint_selection_key(row(0.8, 0.3)) > checkpoint_selection_key(row(0.7, 0.9))
    assert checkpoint_selection_key(row(0.8, 0.6)) > checkpoint_selection_key(row(0.8, 0.5))
    assert checkpoint_selection_key(row(0.8, 0.6, cr=0)) > checkpoint_selection_key(row(0.8, 0.6))
    assert checkpoint_selection_key(row(0.8, 0.6, progress=0.8)) > checkpoint_selection_key(row(0.8, 0.6))
    assert checkpoint_selection_key(row(0.8, 0.6, update=250)) > checkpoint_selection_key(row(0.8, 0.6, update=500))


def test_navigation_validation_restores_rng_and_training_mode(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    seen = []

    def rollout(record, config, model, wrapped, index, inference_seed):
        assert not model.training
        seen.append((wrapped, inference_seed))
        torch.manual_seed(inference_seed)
        torch.rand(3)
        return {"included_in_denominator": True, "episode_id": "val:a", "success": 0,
                "collision": 0, "spl": 0, "stuck": 0, "route_failure": 1, "goal_progress": 0}

    monkeypatch.setattr(train_transfer, "rollout", rollout)
    model = torch.nn.Linear(1, 1).train()
    state = torch.get_rng_state().clone()
    macro = train_transfer.navigation_validate(model, None, [{}], True)
    assert model.training and torch.equal(state, torch.get_rng_state())
    assert seen == [(True, 20260914)]
    assert macro["sr"] == 0 and macro["route_failure_rate"] == 1


def test_checkpoint_and_metadata_publication(tmp_path):
    from tartan.research_score.artifacts import publish
    import json

    checkpoint = tmp_path / "model.pt"
    publish({"model": torch.tensor([1.0, 2.0])}, checkpoint)
    assert torch.equal(torch.load(checkpoint, weights_only=True)["model"], torch.tensor([1.0, 2.0]))
    metadata = tmp_path / "metrics.json"
    publish({"sr": 0.5}, metadata, is_json=True)
    assert json.loads(metadata.read_text()) == {"sr": 0.5}
    assert {path.name for path in tmp_path.iterdir()} == {"model.pt", "metrics.json"}


def test_generated_run_id_and_output_name_rules():
    import re
    import pytest
    from tartan.research_score.artifacts import new_run_id, validate_output_name

    assert re.fullmatch(r"\d{8}T\d{6}Z_transfer_seed11", new_run_id("transfer_seed11"))
    assert validate_output_name("navigation_best.pt") == "navigation_best.pt"
    for name in ["乱码", "bad name", "bad/name", "\ufffd", "..", ""]:
        with pytest.raises(ValueError):
            validate_output_name(name)


def test_proxy_rng_stream_resume_preserves_next_noise_and_sampler():
    import random
    import numpy as np
    streams = {name: torch.Generator().manual_seed(seed) for name, seed in
               (('base_sampler', 11), ('base_noise', 11), ('pair_sampler', 12),
                ('pair_noise', 13), ('id_dropout', 14))}
    saved = train_transfer.capture_rng(streams)
    def draw():
        return (random.random(), np.random.rand(), torch.rand(2),
                {k: torch.rand(3, generator=g) for k, g in streams.items()})
    continuous = draw()
    train_transfer.restore_rng(saved, streams)
    resumed = draw()
    assert continuous[:2] == resumed[:2]
    assert torch.equal(continuous[2], resumed[2])
    assert all(torch.equal(continuous[3][k], resumed[3][k]) for k in streams)
    train_transfer.restore_rng(saved, streams)


def test_proxy_resume_comparison_checks_noise_and_ignores_runtime(tmp_path, monkeypatch):
    import json
    import pytest

    left = {'config': {'cli': {'output': 'continuous', 'smoke': 'cpu'}},
            'model': {}, 'optimizer': {}, 'scheduler': {}, 'scaler': {}, 'rng': {},
            'input_identity': {'sample_ids': ['real-window-id']},
            'training_state': {'elapsed_seconds': 1., 'history': [{'step_s': 2.}],
                               'last_noise': {'base_t': torch.tensor([0.2]),
                                              'base_noise': torch.tensor([1., 2.])}}}
    import copy
    right = copy.deepcopy(left)
    right['config']['cli']['output'] = 'resumed'
    right['training_state']['elapsed_seconds'] = 9.
    continuous, resumed = tmp_path / 'continuous.pt', tmp_path / 'resumed.pt'
    output = tmp_path / 'checks.json'
    torch.save(left, continuous); torch.save(right, resumed)
    monkeypatch.setattr(train_transfer.sys, 'argv', ['train_transfer', '--compare-checkpoints',
                         str(continuous), str(resumed), '--comparison-output', str(output)])
    train_transfer.compare_checkpoints()
    assert json.loads(output.read_text())['status'] == 'PASS'
    right['training_state']['last_noise']['base_noise'][0] += 0.1
    torch.save(right, resumed)
    with pytest.raises(AssertionError):
        train_transfer.compare_checkpoints()


def test_metadata_replaces_warm_file_without_truncation(tmp_path):
 from tartan.research_score import artifacts
 p=tmp_path/'record.json'
 artifacts.publish({'actual':'long first content'},p,True)
 assert p.read_text()
 artifacts.publish({'actual':'record'},p,True)
 assert p.read_text().strip()=='{\n  "actual": "record"\n}'


def test_metadata_refreshes_observed_stale_page_once(tmp_path, monkeypatch):
    import pytest
    from pathlib import Path
    from tartan.research_score import artifacts
    p=tmp_path/'record.json'
    read_bytes=Path.read_bytes
    reads=[];refreshes=[]
    def stale_read(path):
        content=read_bytes(path)
        if path==p:
            reads.append(content)
            if len(reads)==1:return bytes(len(content))
        return content
    monkeypatch.setattr(Path,'read_bytes',stale_read)
    monkeypatch.setattr(artifacts.os,'posix_fadvise',lambda *args:refreshes.append(args))
    artifacts.publish({'actual':'record'},p,True)
    assert len(reads)==2 and len(refreshes)==1
    assert p.read_text().strip()=='{\n  "actual": "record"\n}'
    monkeypatch.setattr(Path,'read_bytes',lambda path:b'corrupt')
    with pytest.raises(OSError,match='metadata publication mismatch'):
        artifacts.publish({'actual':'changed'},p,True)
    assert len(refreshes)==2


def test_five_epoch_validation_uses_complete_pass_not_update_count():
    state={'epoch':5,'cursor':100,'permutation':list(range(101)), 'update':250,'next_val':250}
    cfg={'val_every_epochs':5}
    assert not train_transfer.validation_due(state,cfg,True)
    state['cursor']=101
    assert train_transfer.validation_due(state,cfg,True)
    state['epoch']=6
    assert not train_transfer.validation_due(state,cfg,True)
    assert train_transfer.validation_due(state,cfg,False)


def test_observation_heading_is_not_replaced_by_strafe_direction():
    import numpy as np
    from tartan.research_score.scripts.evaluate_navigation import controller_heading
    # Mathematical controller check, never a saved training fixture.
    sideways=np.column_stack((np.zeros(80),-np.arange(1,81)/10.,np.ones(80),np.zeros(80)))
    assert controller_heading(sideways,1.)==0.
    turning=np.column_stack((np.arange(1,81)/10.,np.zeros(80),np.zeros(80),np.ones(80)))
    assert np.isclose(controller_heading(turning,1.),np.pi/2)
