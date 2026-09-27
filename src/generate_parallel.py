#!/usr/bin/env python3
"""Run disjoint test Source-1 shards against a completed shared FTS5 index."""
import argparse, csv, os, subprocess, sys, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'output')
    ap.add_argument('--threshold',type=float,default=None,help='defaults to work/threshold.txt')
    ap.add_argument('--workers',type=int,default=min(os.cpu_count() or 1,8))
    a=ap.parse_args(); root=Path(__file__).resolve().parents[1]; work=a.work.resolve(); data=a.data.resolve(); output=a.output.resolve()
    workers=max(1,a.workers); shardroot=work/'shards'; shardroot.mkdir(parents=True,exist_ok=True)
    if a.threshold is None:
        a.threshold=float((work/'threshold.txt').read_text(encoding='utf-8').strip())
    procs=[]; logs=[]
    for i in range(workers):
        shardout=shardroot/str(i)/'output'; shardout.mkdir(parents=True,exist_ok=True)
        logpath=shardroot/f'worker_{i}.log'; log=open(logpath,'w',encoding='utf-8')
        cmd=[sys.executable,str(root/'src'/'pipeline.py'),'--data',str(data),'--work',str(work),'--output',str(shardout),'--threshold',str(a.threshold),'--reuse-index','--shard-index',str(i),'--shard-count',str(workers)]
        procs.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)); logs.append(log)
    last_report=0
    while True:
        alive=sum(p.poll() is None for p in procs)
        now=time.time()
        if now-last_report>=30:
            completed=workers-alive
            print(f'Workers complete: {completed}/{workers}',flush=True)
            for i in range(workers):
                mp=shardroot/str(i)/'output'/'matching_results.tsv'
                if mp.exists(): print(f'  shard {i}: {mp.stat().st_size:,} matching bytes',flush=True)
            last_report=now
        if alive==0: break
        time.sleep(5)
    for log in logs: log.close()
    failures=[(i,p.returncode) for i,p in enumerate(procs) if p.returncode]
    if failures:
        for i,_ in failures: print((shardroot/f'worker_{i}.log').read_text(encoding='utf-8',errors='replace')[-4000:])
        raise SystemExit(f'Worker failures: {failures}')
    output.mkdir(parents=True,exist_ok=True)
    specs=(('matching_results.tsv','matched_entity_ids'),('candidate_pairs.tsv','candidate_entity_ids'))
    for filename,second in specs:
        final=output/filename
        with open(final,'w',encoding='utf-8',newline='') as out:
            writer=csv.writer(out,delimiter='\t',lineterminator='\n'); writer.writerow(['source1_entity_id',second])
            for i in range(workers):
                part=shardroot/str(i)/'output'/filename
                if not part.is_file(): raise FileNotFoundError(part)
                with open(part,'r',encoding='utf-8',newline='') as src:
                    reader=csv.reader(src,delimiter='\t'); next(reader,None)
                    for row in reader: writer.writerow(row)
    print(f'Combined all shard outputs under {output}',flush=True)
if __name__=='__main__': main()
