#!/usr/bin/env python3
"""Disk-backed, training-only entity resolution baseline for the ML Challenge 2026."""
from __future__ import annotations
import argparse, csv, hashlib, os, re, sqlite3, sys, time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {"the", "and", "of", "at", "near", "to", "in", "on", "for", "private", "limited", "company", "corporation", "incorporated", "inc", "ltd", "pvt", "corp", "llc", "llp"}
K_PER_SOURCE = 20
K_FALLBACK_PER_SOURCE = 10
FINAL_CANDIDATES_PER_SOURCE = 40
DB_BATCH = 5000

def norm(s):
    return " ".join(TOKEN_RE.findall((s or "").lower()))

def tokens(s):
    return list(dict.fromkeys(t for t in TOKEN_RE.findall((s or "").lower()) if len(t) > 1 and t not in STOP))

def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")

def create_index(db, table, path):
    db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute(f"CREATE VIRTUAL TABLE {table} USING fts5(entity_id UNINDEXED,country UNINDEXED,business_name,business_address, tokenize='unicode61 remove_diacritics 2')")
    insert = f"INSERT INTO {table}(entity_id,country,business_name,business_address) VALUES(?,?,?,?)"
    batch=[]
    for row in read_rows(path):
        batch.append((row["entity_id"], row.get("country", ""), row.get("business_name", ""), row.get("business_address", "")))
        if len(batch) >= DB_BATCH:
            db.executemany(insert,batch); db.commit(); batch.clear()
    if batch: db.executemany(insert,batch); db.commit()

def query_candidates(db, table, row):
    # Use several independent lexical blocks. A single token is often corrupted
    # or too common, so relying on only the longest token creates an avoidable
    # recall ceiling before the matcher sees the record.
    terms={}
    for field in ("business_name", "business_address"):
        ts=tokens(row.get(field,""))
        if ts:
            # Prefer long terms, but retain several independent signals for
            # typos, abbreviations, and records with partial fields.
            terms[field]=sorted(ts, key=lambda t: (len(t)<3, -len(t), t))[:3]
    found={}
    searches=[]
    if "business_name" in terms and "business_address" in terms:
        for name_term in terms["business_name"][:2]:
            for address_term in terms["business_address"][:2]:
                searches.append((f'business_name:"{name_term}" AND business_address:"{address_term}"',8))
    for field, field_terms in terms.items():
        searches.append((" OR ".join(f'{field}:"{term}"' for term in field_terms),24))
    for q,limit in searches:
        try:
            sql=f"SELECT entity_id,business_name,business_address,country FROM {table} WHERE {table} MATCH ? LIMIT ?"
            for rec in db.execute(sql,(q,limit)):
                found[rec[0]]=rec[:4]
        except sqlite3.OperationalError:
            pass
    # Keep inference bounded. Rank by independent lexical evidence rather than
    # SQLite insertion order, which is not a useful relevance signal.
    ranked=[]
    for rec in found.values():
        name_set=set(tokens(rec[1])); address_set=set(tokens(rec[2]))
        name_hits=len(name_set & set(terms.get("business_name", [])))
        address_hits=len(address_set & set(terms.get("business_address", [])))
        ranked.append((-(2*name_hits+address_hits), -name_hits, -address_hits, rec[0], rec))
    ranked.sort()
    return [item[-1] for item in ranked[:60]]

def pair_features(a,b):
    an, bn = norm(a.get("business_name")), norm(b.get("business_name"))
    aa, ba = norm(a.get("business_address")), norm(b.get("business_address"))
    at,bt=set(tokens(an)),set(tokens(bn)); ad,bd=set(tokens(aa)),set(tokens(ba))
    jn=len(at&bt)/max(1,len(at|bt)); ja=len(ad&bd)/max(1,len(ad|bd))
    # Edit distance is useful only after token blocking has found some shared
    # evidence. Skip expensive character alignment for unrelated fields.
    rn=SequenceMatcher(None,an,bn).ratio() if an and bn and jn>=0.10 else jn
    ra=SequenceMatcher(None,aa,ba).ratio() if aa and ba and ja>=0.10 else ja
    # A high name similarity plus independent address evidence is required for a positive.
    same_country=float(bool(a.get("country")) and bool(b.get("country")) and a.get("country").strip().casefold()==b.get("country").strip().casefold())
    country_conflict=float(bool(a.get("country")) and bool(b.get("country")) and not same_country)
    # Address digits (house numbers, PIN/ZIP codes) are high-value corroborating
    # evidence. Compare them separately because token Jaccard dilutes them.
    adigits=set(re.findall(r"\d+",aa)); bdigits=set(re.findall(r"\d+",ba))
    digit_overlap=len(adigits & bdigits)/max(1,len(adigits | bdigits))
    return (rn,jn,ra,ja,same_country,country_conflict,digit_overlap)

def score(f):
    rn,jn,ra,ja,same_country,country_conflict,digit_overlap=f
    # Blend character and token evidence instead of letting either one dominate.
    # Name is the strongest field; address digits and country corroborate it.
    base=0.68*(0.72*rn+0.28*jn)+0.32*(0.65*ra+0.25*ja+0.10*digit_overlap)
    if same_country: base+=0.025
    if country_conflict: base-=0.08
    return max(0.0,min(1.0,base))

def f05(pred, truth):
    if not pred and not truth: return 1.0
    if not pred or not truth: return 0.0
    tp=len(pred&truth); p=tp/len(pred); r=tp/len(truth)
    return 1.25*p*r/(0.25*p+r) if (0.25*p+r) else 0.0

def split_validation(s1_id):
    return int(hashlib.md5(s1_id.encode()).hexdigest()[:8],16)%5 == 0

def tune_threshold(db, source_paths, ground_truth_path, limit=None):
    truth={}
    for row in read_rows(ground_truth_path):
        truth[row["source1_entity_id"]]=set(filter(None,(row.get("matched_entity_ids") or "").split(",")))
    thresholds=[i/100 for i in range(30,96,2)]
    totals=[0.0]*len(thresholds); n=0
    for i,row in enumerate(read_rows(source_paths[0])):
        sid=row["entity_id"]
        if not split_validation(sid): continue
        options=[]
        for tbl in ("train_s2","train_s3"):
            for cand in query_candidates(db,tbl,row):
                f=pair_features(row,{"business_name":cand[1],"business_address":cand[2],"country":cand[3]})
                options.append((cand[0],score(f)))
        gold=truth.get(sid,set())
        for j,t in enumerate(thresholds):
            totals[j]+=f05({x for x,s in options if s>=t},gold)
        n+=1
        if limit and n>=limit: break
        if n%10000==0: print(f"validation entities: {n}",flush=True)
    if not n: raise RuntimeError("No validation rows found")
    vals=[x/n for x in totals]
    best=max(range(len(vals)),key=lambda j:(vals[j],thresholds[j]))
    return thresholds[best], vals[best], n

def write_outputs(root, db, threshold, out, shard_index=0, shard_count=1):
    out.mkdir(parents=True,exist_ok=True)
    match_path=out/"matching_results.tsv"; cand_path=out/"candidate_pairs.tsv"
    with open(match_path,"w",encoding="utf-8",newline="") as mf, open(cand_path,"w",encoding="utf-8",newline="") as cf:
        mw=csv.writer(mf,delimiter="\t",lineterminator="\n"); cw=csv.writer(cf,delimiter="\t",lineterminator="\n")
        mw.writerow(["source1_entity_id","matched_entity_ids"]); cw.writerow(["source1_entity_id","candidate_entity_ids"])
        for ix,row in enumerate(read_rows(root/"dataset/test/test_source1.tsv"),1):
            if int(hashlib.md5(row["entity_id"].encode()).hexdigest()[:8],16) % shard_count != shard_index:
                continue
            candidates=[]; chosen=[]
            for tbl in ("test_s2","test_s3"):
                for c in query_candidates(db,tbl,row):
                    f=pair_features(row,{"business_name":c[1],"business_address":c[2],"country":c[3]})
                    candidates.append((c[0],score(f)))
            candidates.sort(key=lambda x:(-x[1],x[0]))
            ids=list(dict.fromkeys(x[0] for x in candidates))
            chosen=[x[0] for x in candidates if x[1]>=threshold]
            mw.writerow([row["entity_id"],",".join(dict.fromkeys(chosen))]); cw.writerow([row["entity_id"],",".join(ids)])
            if ix%10000==0: mf.flush(); cf.flush(); print(f"test entities: {ix}",flush=True)
    print(f"Wrote {match_path} and {cand_path}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True,help="student_resource directory containing dataset/train and dataset/test")
    ap.add_argument("--work",type=Path,default=Path("work"),help="temporary SQLite index directory")
    ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"output",help="submission output directory")
    ap.add_argument("--validation-limit",type=int,default=None,help="optional cap for a quick smoke run; omit for full validation")
    ap.add_argument("--threshold",type=float,default=None,help="skip threshold tuning and use this threshold")
    ap.add_argument("--reuse-index",action="store_true",help="reuse completed SQLite indexes in --work")
    ap.add_argument("--prepare-only",action="store_true",help="build indexes/tune threshold and stop before inference")
    ap.add_argument("--shard-index",type=int,default=0,help="worker partition number, from 0 to shard-count-1")
    ap.add_argument("--shard-count",type=int,default=1,help="number of disjoint output partitions")
    args=ap.parse_args(); data=args.data.resolve(); work=args.work.resolve(); work.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(work/"entity_index.sqlite"))
    train=data/"dataset/train"; test=data/"dataset/test"
    if not args.reuse_index:
        for tbl,path in (("train_s2",train/"train_source2.tsv"),("train_s3",train/"train_source3.tsv"),("test_s2",test/"test_source2.tsv"),("test_s3",test/"test_source3.tsv")):
            if not path.is_file(): raise FileNotFoundError(path)
            print(f"Indexing {path.name}",flush=True); create_index(db,tbl,path)
            print(f"Compacting {tbl}",flush=True)
            db.execute(f"INSERT INTO {tbl}({tbl}) VALUES('optimize')"); db.commit()
    else:
        expected={"train_s2","train_s3","test_s2","test_s3"}
        available={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not expected.issubset(available): raise RuntimeError(f"Incomplete index database; missing {sorted(expected-available)}")
    if args.threshold is None:
        threshold,val,n=tune_threshold(db,(train/"train_source1.tsv",),train/"train_ground_truth.tsv",args.validation_limit)
        print(f"Validation macro F0.5={val:.5f} over {n} Source 1 records; selected threshold={threshold:.2f}",flush=True)
    else: threshold=args.threshold
    (work/"threshold.txt").write_text(f"{threshold:.2f}\n",encoding="utf-8")
    if args.prepare_only:
        db.close(); return
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count: raise ValueError("invalid shard index/count")
    write_outputs(data,db,threshold,args.output.resolve(),args.shard_index,args.shard_count); db.close()
if __name__=="__main__": main()




