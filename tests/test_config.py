from tpv import config


def test_experiment_count_and_shape():
    assert set(config.EXPERIMENTS.keys()) == {0, 1, 2, 3, 4, 5}
    for exp, classes in config.EXPERIMENTS.items():
        assert len(classes) == 2, f"exp {exp} must have 2 classes"
        labels = sorted(c["label"] for c in classes)
        assert labels == [0, 1]
        for c in classes:
            assert set(c.keys()) == {"runs", "annotations", "label"}
            assert all(r in range(3, 15) for r in c["runs"])
            assert all(a in {"T1", "T2"} for a in c["annotations"])


def test_exp0_is_t1_vs_t2_same_runs():
    classes = config.EXPERIMENTS[0]
    assert classes[0]["runs"] == [3, 7, 11] and classes[0]["annotations"] == ["T1"]
    assert classes[1]["runs"] == [3, 7, 11] and classes[1]["annotations"] == ["T2"]


def test_exp4_is_real_vs_imagined_pooled():
    classes = config.EXPERIMENTS[4]
    assert classes[0]["runs"] == [3, 7, 11] and classes[0]["annotations"] == ["T1", "T2"]
    assert classes[1]["runs"] == [4, 8, 12] and classes[1]["annotations"] == ["T1", "T2"]


def test_run_to_experiment_unique_mapping():
    assert config.RUN_TO_EXPERIMENT[14] == 3
    assert config.RUN_TO_EXPERIMENT[3] == 0
    assert config.RUN_TO_EXPERIMENT[8] == 1
    assert config.RUN_TO_EXPERIMENT[9] == 2
    assert sorted(config.RUN_TO_EXPERIMENT.keys()) == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert set(config.RUN_TO_EXPERIMENT.values()) == {0, 1, 2, 3}


def test_runs_for_experiment():
    assert sorted(config.runs_for_experiment(0)) == [3, 7, 11]
    assert sorted(config.runs_for_experiment(4)) == [3, 4, 7, 8, 11, 12]


def test_get_seed_reads_env(monkeypatch):
    # get_seed() reads os.environ live, so no module reload is needed.
    monkeypatch.delenv("TPV_SEED", raising=False)
    assert config.get_seed() is None
    monkeypatch.setenv("TPV_SEED", "42")
    assert config.get_seed() == 42
