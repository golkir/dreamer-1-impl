import pytest

from dreamer.config import Config, make_config, parse_overrides, set_field


def test_presets_apply():
    atari = make_config("atari")
    assert atari.env.suite == "atari"
    assert atari.env.action_repeat == 4
    assert atari.train.use_continue
    assert make_config("dmc").train.batch_length == 50


def test_overrides_are_typed():
    config = make_config(
        "dmc",
        parse_overrides(
            [
                "train.batch_size=16",
                "train.model_lr=1e-3",
                "train.steps=2e6",
                "train.use_continue=true",
                "env.size=(32, 32)",
                "env.task=cheetah_run",
                "train.free_nats=3",
            ]
        ),
    )
    assert config.train.batch_size == 16
    assert config.train.model_lr == pytest.approx(1e-3)
    assert config.train.steps == 2_000_000 and isinstance(config.train.steps, int)
    assert config.train.use_continue is True
    assert config.env.size == (32, 32)
    assert config.env.task == "cheetah_run"
    assert isinstance(config.train.free_nats, float)


def test_bad_overrides_raise():
    config = Config()
    with pytest.raises(KeyError):
        set_field(config, "train.nope", 1)
    with pytest.raises(TypeError):
        set_field(config, "train.batch_size", "abc")
    with pytest.raises(ValueError):
        make_config("unknown")


def test_dict_roundtrip():
    config = make_config("atari", {"run.seed": 3})
    assert Config.from_dict(config.to_dict()) == config
