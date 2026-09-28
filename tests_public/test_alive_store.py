import tempfile
from pathlib import Path
from kev.uc51a3.alive import AliveStore

def test_fact_revision_and_ledger_without_model_weights():
    with tempfile.TemporaryDirectory() as d:
        store=AliveStore(Path(d))
        first=store.remember_fact("project codename","Lumen")
        second=store.remember_fact("project codename","Nova",correction=True)
        assert first["revision"]==1
        assert second["revision"]==2
        assert second["supersedes"]=="Lumen"
        assert store.verify_ledger()["valid"] is True
