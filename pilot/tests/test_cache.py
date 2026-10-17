"""Step reuse must be invalidated when an upstream step's output changes."""
import json

from speech_pilot.config import load_config
from speech_pilot.pipeline import PilotRun


def test_upstream_change_invalidates_downstream(tmp_path):
    cfg = load_config(overrides={"run.output_root": str(tmp_path)})
    cfg.run.output_root = tmp_path
    run = PilotRun(cfg, "t")
    for step, content in (("01_inspect", "a"), ("02_prepare", "b")):
        (run._dir(step) / "out.json").write_text(content)
        run._finish(step)
    assert PilotRun(cfg, "t")._is_cached("02_prepare")
    # recompute 01 with a different output: 02 must no longer count as cached
    (run.dir / "01_inspect" / "out.json").write_text("changed")
    run._finish("01_inspect")
    assert not PilotRun(cfg, "t")._is_cached("02_prepare")
    assert json.loads((run.dir / "01_inspect" / "step.json").read_text())["output_hash"]
