import gzip
import zipfile

import pandas as pd
import pytest

from spsbert.cli import main as cli_main
from spsbert.data import build_dataset
from spsbert.config import DataConfig
from spsbert.sqldump import _Incomplete, _unescape, convert, parse_values

DUMP = r"""-- MySQL dump 10.13
/*!40101 SET NAMES utf8mb4 */;
DROP TABLE IF EXISTS `Issue`;
CREATE TABLE `Issue` (
  `ID` int NOT NULL,
  `Issue_Key` varchar(255) DEFAULT NULL,
  `Title` text,
  `Description_Text` longtext,
  `Creation_Date` datetime DEFAULT NULL,
  `Story_Point` float DEFAULT NULL,
  `Project_ID` int DEFAULT NULL,
  PRIMARY KEY (`ID`),
  KEY `fk_project` (`Project_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
LOCK TABLES `Issue` WRITE;
INSERT INTO `Issue` VALUES (1,'P-1','Fix \'quote\', (paren) and ; semicolon','line1\nline2, with comma and ''double'' and back\\slash','2020-01-01 10:00:00',3,1),(2,'P-2','no desc',NULL,'2020-01-02 00:00:00',NULL,1),(3,'Q-1','neg and float','z',NULL,5.5,2);
INSERT INTO `Issue` VALUES (4,'Q-2','spans
two lines','desc','2021-05-05',8,2);
INSERT INTO `Issue` VALUES (5,'Z-1','orphan project',NULL,NULL,2,99);
UNLOCK TABLES;
DROP TABLE IF EXISTS `Project`;
CREATE TABLE `Project` (
  `ID` int NOT NULL,
  `Description` text,
  `Name` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB;
INSERT INTO `Project` VALUES (1,'first','Alpha, Inc'),(2,NULL,'C++ driver (Beta)');
"""


def _write(tmp_path, name="d.sql", text=DUMP):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def check_output(csv):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert list(df.columns) == ["Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date", "Story_Point"]
    assert list(df.Issue_Key) == ["P-1", "Q-1", "Q-2"]                    # NULL story point and orphan project dropped
    assert df.Project_Name.tolist() == ["Alpha, Inc", "C++ driver (Beta)", "C++ driver (Beta)"]
    assert df.Title[0] == "Fix 'quote', (paren) and ; semicolon"
    assert df.Description_Text[0] == "line1\nline2, with comma and 'double' and back\\slash"
    assert df.Title[2] == "spans\ntwo lines"                              # multi-line statement
    assert df.Story_Point.tolist() == ["3", "5.5", "8"]


def test_convert_plain_dump(tmp_path):
    st = convert(_write(tmp_path), tmp_path / "o.csv")
    assert st == {"issue_rows": 5, "issues_with_story_point": 4, "issues_unknown_project": 1, "written": 3, "projects": 2}
    check_output(tmp_path / "o.csv")


def test_convert_gz_and_zip_and_per_row_inserts(tmp_path):
    p = _write(tmp_path)
    with gzip.open(tmp_path / "d.sql.gz", "wb") as f:
        f.write(p.read_bytes())
    convert(tmp_path / "d.sql.gz", tmp_path / "g.csv"); check_output(tmp_path / "g.csv")
    with zipfile.ZipFile(tmp_path / "d.zip", "w") as z:
        z.writestr("TAWOS.sql", p.read_text(encoding="utf-8"))
    convert(tmp_path / "d.zip", tmp_path / "z.csv"); check_output(tmp_path / "z.csv")
    row_by_row = DUMP.replace("),(", ");\nINSERT INTO `Issue` VALUES (", 1).replace("),(1,'first'", "),(1,'first'")
    assert row_by_row.count("INSERT INTO `Issue`") == DUMP.count("INSERT INTO `Issue`") + 1                      # the first statement was really split
    convert(_write(tmp_path, "r.sql", row_by_row), tmp_path / "r.csv"); check_output(tmp_path / "r.csv")


def test_insert_with_explicit_column_list_and_column_aliases(tmp_path):
    dump = ("CREATE TABLE `Project` (\n  `ID` int,\n  `Name` text\n) ENGINE=InnoDB;\n"
            "INSERT INTO `Project` VALUES (7,'X');\n"
            "INSERT INTO `Issue` (`Project_ID`,`Issue_Key`,`Summary`,`Story_Points`) VALUES (7,'X-1','t',2),(7,'X-2','u',NULL);\n")
    st = convert(_write(tmp_path, "a.sql", dump), tmp_path / "a.csv")
    df = pd.read_csv(tmp_path / "a.csv")
    assert st["written"] == 1 and df.Issue_Key.tolist() == ["X-1"] and df.Story_Point.tolist() == [2]


def test_parser_primitives():
    assert parse_values("(1,'a''b',NULL,-2.5e3,0x4142,_binary 'x\\ny');", 0) == [["1", "a'b", None, "-2.5e3", "AB", "x\ny"]]
    assert parse_values("(1),(2);", 0) == [["1"], ["2"]]
    assert _unescape(r"a\'b\\c\n") == "a'b\\c\n"
    with pytest.raises(_Incomplete):
        parse_values("(1,'unterminated", 0)
    with pytest.raises(ValueError):
        parse_values("(1,@bad);", 0)


def test_errors_are_clear(tmp_path):
    with pytest.raises(ValueError, match="No INSERT INTO `Issue`"):
        convert(_write(tmp_path, "e.sql", "CREATE TABLE `Other` (\n `a` int\n);\n"), tmp_path / "o.csv")
    trunc = DUMP.split("INSERT INTO `Issue` VALUES (4")[0] + "INSERT INTO `Issue` VALUES (4,'cut"
    with pytest.raises(ValueError, match="truncated"):
        convert(_write(tmp_path, "t.sql", trunc), tmp_path / "o.csv")
    nocol = "CREATE TABLE `Issue` (\n  `ID` int,\n  `Foo` text\n);\nINSERT INTO `Issue` VALUES (1,'a');\n"
    with pytest.raises(ValueError, match="cannot find column"):
        convert(_write(tmp_path, "n.sql", nocol), tmp_path / "o.csv")


def test_cli_and_end_to_end_into_build_dataset(tmp_path, capsys):
    p = _write(tmp_path)
    assert cli_main(["sql2csv", str(p), "--out", str(tmp_path / "c.csv")]) == 0
    assert "Wrote 3 issues" in capsys.readouterr().out
    df, rep = build_dataset(DataConfig(csv_path=str(tmp_path / "c.csv"), min_issues=1, projects=["alpha, inc"]))
    assert df.project.tolist() == ["Alpha, Inc"] and df.story_point.tolist() == [3.0]


def test_project_name_suggestions(tmp_path):
    rows = [["Alpha Proj", f"K-{i}", f"t{i}", "d", "2020", 2] for i in range(30)]
    csv = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=["Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date", "Story_Point"]).to_csv(csv, index=False)
    _, rep = build_dataset(DataConfig(csv_path=str(csv), projects=["Alpha Proj", "Alpa Proj"], min_issues=20))
    assert rep["requested_projects_suggestions"] == {"Alpa Proj": ["Alpha Proj"]}
