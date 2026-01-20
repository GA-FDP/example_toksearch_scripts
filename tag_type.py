from toksearch import Pipeline
from toksearch.sql.mssql import connect_d3drdb



with connect_d3drdb(db="code_rundb") as conn:
    query = "select shot, runtag, tree from plasmas where tree like '%EFIT%' and runtag = 'JT_TS'"

    pipeline = Pipeline.from_sql(conn, query)

    results = pipeline.compute_serial()

    print(len(results))

    for result in results:
        print(result)
