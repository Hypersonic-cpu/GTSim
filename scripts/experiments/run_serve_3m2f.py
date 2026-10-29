from pathlib import Path
import json,subprocess,time,re
root=Path(__file__).resolve().parents[2];out=root/'validation/results/serve-3m2f';out.mkdir(parents=True,exist_ok=True)
configs={'normal':root/'configs/v0.1.0/a100-pcie40-simple.yaml','hbf':root/'configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml'}
cases=[('fa2',None),('gqa_qkvgate',(1,10240,4096)),('gdn_qkvz',(1,12288,4096)),('output_proj',(1,4096,4096)),('mlp_gate_up',(1,24576,4096)),('mlp_down',(1,4096,12288))]
rows=[]
for arch,cfg in configs.items():
 for name,shape in cases:
  stem=out/(arch+'-'+name)
  cmd=[str(root/'build/gtsim_fa2'),str(cfg),'1','16','4','1','32768','256','47'] if name=='fa2' else [str(root/'build/gtsim_gemm'),str(cfg),*map(str,shape),'--bm','64','--bn','128','--bk','64','--dtype','bf16']
  stem.with_suffix('.command.json').write_text(json.dumps(cmd,indent=2));t=time.monotonic()
  with stem.with_suffix('.log').open('w') as f,stem.with_suffix('.resources').open('w') as e:r=subprocess.run(['/usr/bin/time','-l',*cmd],stdout=f,stderr=e)
  log=stem.with_suffix('.log').read_text();resources=stem.with_suffix('.resources').read_text();m=re.search(r'(\d+)\s+maximum resident set size',resources);cy=re.search(r'^Total cycles: (\d+)',log,re.M)
  row={'arch':arch,'operator':name,'returncode':r.returncode,'wall_seconds':time.monotonic()-t,'peak_rss_bytes':int(m[1]) if m else None,'cycles':int(cy[1]) if cy else None}
  if cy:row['us']=int(cy[1])/1410
  row['hbf_read_bytes']=int(re.search(r'HBF requests:.*?bytes: read (\d+)',log)[1]);row['hbf_write_bytes']=int(re.search(r'HBF requests:.*?write (\d+), average',log)[1])
  if name=='fa2':assert row['hbf_read_bytes']==row['hbf_write_bytes']==0
  if arch=='hbf' and shape:assert row['hbf_read_bytes']>0 and row['hbf_write_bytes']==0
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
  if r.returncode:raise SystemExit(r.returncode)
