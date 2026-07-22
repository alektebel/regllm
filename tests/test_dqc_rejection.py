from training.dq import checks_db


def test_rejected_check_cannot_be_validated(tmp_path):
    conn = checks_db.connect(tmp_path / "checks.db")
    try:
        check_id = checks_db.insert_check(
            conn,
            name="test",
            description="test",
            severity="MED",
            category="rango",
            sql="SELECT ID_CONTR_CICLO_LGD FROM ciclos",
        )
        assert checks_db.set_status(conn, check_id, "rejected")
        assert not checks_db.set_status(conn, check_id, "validated")
        assert checks_db.get_check(conn, check_id)["status"] == "rejected"
    finally:
        conn.close()
