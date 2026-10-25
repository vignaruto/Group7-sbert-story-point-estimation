"""Assignment short names must map onto the 39 real TAWOS project names (taken from the real dump, Kaggle run)."""
import pandas as pd

from spsbert.config import DataConfig
from spsbert.data import build_dataset, read_aliases, read_project_list
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_TAWOS = """Lsstcorp Data management|The MongoDB Engineering|The Titanium SDK|Mule|Spring XD|Appcelerator Studio|Apache Mesos|DotNetNuke Platform|Sonatype Nexus|Moodle|Titanium Mobile Platform|Hyperledger Sawtooth|Aptana Studio|MongoDB Core Server|Hyperledger Indy SDK|Hyperledger Indy Node|Lyrasis Dura Cloud|Atlassian Confluence Server|Hyperledger Fabric|Atlassian Bamboo|MongoDB Compass|Apache Usergrid|Mule APIkit|Atlassian Clover|Atlassian Jira Server|Command-Line Interface|Hyperledger Blockchain Explorer|Atlassian Jira Cloud|Atlassian Confluence Cloud|Atlassian Atlassian Software Server|Atlassian Software Cloud|Alloy Framework|Spring DataCass|Appcelerator Daemon|Atlassian FishEye|The Mongo Java driver|The Mongo C++ driver|Atlassian Crowd|Apache MXNet""".split("|")


def test_every_guideline_name_maps_to_exactly_one_distinct_real_tawos_project(tmp_path):
    assert len(REAL_TAWOS) == 39 and len(set(REAL_TAWOS)) == 39
    rows = [[n, f"K-{i}", f"title {n} {i}", "d", "2020", 3] for n in REAL_TAWOS for i in range(25)]
    csv = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=["Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date", "Story_Point"]).to_csv(csv, index=False)
    cfg = DataConfig(csv_path=str(csv), projects_file=str(ROOT / "config" / "guideline_projects.txt"),
                     aliases_file=str(ROOT / "config" / "project_aliases.csv"), min_issues=20)
    df, rep = build_dataset(cfg)
    assert rep["requested_projects_not_found"] == []
    assert df.project.nunique() == 39 and set(df.project) == set(REAL_TAWOS)       # all 39, none twice
    assert len(rep["inferred_name_mappings"]) == 2                                  # the two matched by elimination are flagged
    al = read_aliases(ROOT / "config" / "project_aliases.csv")
    assert len(read_project_list(ROOT / "config" / "guideline_projects.txt")) == 39 and len(al) == 26


def test_inferred_name_notice_is_logged_only_once(tmp_path, caplog):
    import logging
    import spsbert.data as d
    d._LOGGED.clear()
    rows = [[n, f"K-{i}", f"title {n} {i}", "x", "2020", 3] for n in ("Aptana Studio", "The MongoDB Engineering") for i in range(25)]
    csv = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=["Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date", "Story_Point"]).to_csv(csv, index=False)
    cfg = DataConfig(csv_path=str(csv), projects_file=str(ROOT / "config" / "guideline_projects.txt"),
                     aliases_file=str(ROOT / "config" / "project_aliases.csv"), min_issues=20)
    with caplog.at_level(logging.WARNING, logger="spsbert.data"):
        for _ in range(3):
            build_dataset(cfg)
    assert sum("ELIMINATION" in r.message for r in caplog.records) == 1
