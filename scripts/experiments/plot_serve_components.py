"""Plot only matched measured/modeled components; never label this TPOT."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.data.read_text());fig,axes=plt.subplots(1,3,figsize=(12,4.5))
scenarios=['oversubscription_trace_gpu','normal','hbf'];labels=['HostRAM trace: GPU kernels only','5M: simulated','3M2F: simulated'];colors=['#b7b7b7','#4986b8','#df9252']
for ax,key,title,scale in zip(axes,['GQA','GDN','decode_component_sum'],['GQA covered components','GDN covered components','32-layer covered components'],[1,1,1000]):
 vals=[d[s][key]/scale for s in scenarios];bars=ax.bar(range(3),vals,color=colors,width=.64)
 ax.set_xticks(range(3),['HostRAM','5M','3M2F']);ax.set_title(title,fontsize=11);ax.set_ylabel('ms' if scale==1000 else 'μs');ax.set_ylim(0,max(vals)*1.2)
 ax.spines[['top','right']].set_visible(False)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v,f'{v:.2f}',ha='center',va='bottom',fontsize=10)
fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],labels,loc='upper center',ncol=3,frameon=False,fontsize=10)
fig.text(.5,.025,'Matched projections + FA2 only. Excludes auxiliary kernels, host stalls and CPU offload transfers. NOT TPOT.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.08,1,.88]);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);fig.savefig(a.output.with_suffix('.svg'))
