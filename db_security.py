import duckdb


def lock_down(con: duckdb.DuckDBPyConnection) -> None:
    """Restrict what generated SQL can do. Call this AFTER all data is loaded.

    - enable_external_access = false  -> read_csv('/etc/passwd'), COPY ... TO, ATTACH,
                                         INSTALL and similar file/network access stop working.
    - lock_configuration = true       -> generated SQL can no longer change any setting,
                                         so it cannot switch the protection back on.
    The order matters: the lock must be the very last SET.
    """
    # Optional extra hardening. Older DuckDB versions may not know this setting,
    # so a failure here is ignored on purpose.
    try:
        con.execute("SET python_enable_replacements = false")
    except duckdb.Error:
        pass

    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")

