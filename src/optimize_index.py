import sqlite3, sys
from pathlib import Path
path=Path(sys.argv[1])
db=sqlite3.connect(str(path))
for table in ('train_s2','train_s3','test_s2','test_s3'):
    print(f'Optimizing {table}', flush=True)
    db.execute(f"INSERT INTO {table}({table}) VALUES('optimize')")
    db.commit()
print('Index optimization complete', flush=True)
db.close()
