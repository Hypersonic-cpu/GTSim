"""Algorithm-only TE definitions; no tiles, warps, pipelines, or intrinsics.

These are functional specifications, not a claim that the existing Hopper
GEMM-only GTSim lowering can compile FA2 or a recurrent GDN graph.
"""
import math
from tvm import te


def fa2(batch=1, heads_q=16, heads_kv=4, seq=32768, dim=256):
    q = te.placeholder((batch, heads_q, dim), 'bfloat16', 'Q')
    k = te.placeholder((batch, heads_kv, seq, dim), 'bfloat16', 'K')
    v = te.placeholder((batch, heads_kv, seq, dim), 'bfloat16', 'V')
    rd = te.reduce_axis((0, dim), 'd')
    raw_score = te.compute((batch, heads_q, seq), lambda b,h,t: te.sum(
        q[b,h,rd].astype('float32') * k[b,h//(heads_q//heads_kv),t,rd].astype('float32'), axis=rd), name='raw_score')
    score = te.compute((batch,heads_q,seq),lambda b,h,t:raw_score[b,h,t]/math.sqrt(dim),name='score')
    rt = te.reduce_axis((0, seq), 't_max')
    maximum = te.compute((batch, heads_q), lambda b,h: te.max(score[b,h,rt], axis=rt), name='row_max')
    exp = te.compute((batch, heads_q, seq), lambda b,h,t: te.exp(score[b,h,t]-maximum[b,h]), name='exp')
    rt = te.reduce_axis((0, seq), 't_sum')
    denom = te.compute((batch, heads_q), lambda b,h: te.sum(exp[b,h,rt], axis=rt), name='row_sum')
    rt = te.reduce_axis((0, seq), 't_out')
    raw_o = te.compute((batch, heads_q, dim), lambda b,h,d: te.sum(
        exp[b,h,rt] * v[b,h//(heads_q//heads_kv),rt,d].astype('float32'), axis=rt), name='raw_o')
    o = te.compute((batch,heads_q,dim),lambda b,h,d:(raw_o[b,h,d]/denom[b,h]).astype('bfloat16'),name='O')
    return [q,k,v,o]


def gdn(batch=1, hidden=4096, nk=16, nv=32, dk=128, dv=128, ff=12288):
    kd,vd = nk*dk,nv*dv
    channels = 2*kd+vd
    x = te.placeholder((batch,hidden),'bfloat16','X')
    wq = te.placeholder((hidden,channels+vd),'bfloat16','W_qkvz')
    wb = te.placeholder((hidden,2*nv),'bfloat16','W_ba')
    wo = te.placeholder((vd,hidden),'bfloat16','W_out')
    wf = te.placeholder((hidden,2*ff),'bfloat16','W_gate_up')
    wd = te.placeholder((ff,hidden),'bfloat16','W_down')
    wc = te.placeholder((channels,4),'bfloat16','W_conv')
    wi = te.placeholder((hidden,),'float32','W_input_norm')
    wp = te.placeholder((hidden,),'float32','W_post_norm')
    wg = te.placeholder((dv,),'float32','W_gated_norm')
    al = te.placeholder((nv,),'float32','A_log')
    dt = te.placeholder((nv,),'float32','dt_bias')
    cs = te.placeholder((batch,channels,3),'bfloat16','ConvState')
    state = te.placeholder((batch,nv,dk,dv),'float32','State')
    eps=1e-6
    def rms(tensor,weight,name):
        r=te.reduce_axis((0,hidden),name+'_r')
        ms=te.compute((batch,),lambda b:te.sum((tensor[b,r].astype('float32')*tensor[b,r].astype('float32')),axis=r),name=name+'_sum')
        return te.compute((batch,hidden),lambda b,d:(tensor[b,d].astype('float32')/te.sqrt(ms[b]/hidden+eps)*(1+weight[d])).astype('bfloat16'),name=name)
    def linear(inp,weight,n,reduction,name):
        r=te.reduce_axis((0,reduction),name+'_k')
        acc=te.compute((batch,n),lambda b,j:te.sum(inp[b,r].astype('float32')*weight[r,j].astype('float32'),axis=r),name=name+'_acc')
        return te.compute((batch,n),lambda b,j:acc[b,j].astype('bfloat16'),name=name)
    norm=rms(x,wi,'input_norm')
    qkvz=linear(norm,wq,channels+vd,hidden,'qkvz')
    ba=linear(norm,wb,2*nv,hidden,'ba')
    conv=te.compute((batch,channels),lambda b,c:cs[b,c,0].astype('float32')*wc[c,0].astype('float32')+cs[b,c,1].astype('float32')*wc[c,1].astype('float32')+cs[b,c,2].astype('float32')*wc[c,2].astype('float32')+qkvz[b,c].astype('float32')*wc[c,3].astype('float32'),name='conv')
    conv_silu=te.compute((batch,channels),lambda b,c:(conv[b,c]/(1+te.exp(-conv[b,c]))).astype('bfloat16'),name='conv_silu')
    q=te.compute((batch,nk,dk),lambda b,h,d:conv_silu[b,h*dk+d].astype('float32'),name='q')
    k=te.compute((batch,nk,dk),lambda b,h,d:conv_silu[b,kd+h*dk+d].astype('float32'),name='k')
    rd=te.reduce_axis((0,dk),'q_l2_d')
    qsum=te.compute((batch,nk),lambda b,h:te.sum((q[b,h,rd]*q[b,h,rd]),axis=rd),name='q_l2')
    rd=te.reduce_axis((0,dk),'k_l2_d')
    ksum=te.compute((batch,nk),lambda b,h:te.sum((k[b,h,rd]*k[b,h,rd]),axis=rd),name='k_l2')
    qn=te.compute((batch,nk,dk),lambda b,h,d:q[b,h,d]/te.sqrt(qsum[b,h]+eps)/math.sqrt(dk),name='q_norm')
    kn=te.compute((batch,nk,dk),lambda b,h,d:k[b,h,d]/te.sqrt(ksum[b,h]+eps),name='k_norm')
    a=te.compute((batch,nv),lambda b,h:ba[b,nv+h].astype('float32')+dt[h],name='a')
    # Stable softplus; plain exp is retained only in the bounded negative term.
    softplus=te.compute((batch,nv),lambda b,h:te.if_then_else(a[b,h]>0,a[b,h],0.0)+te.log(1+te.exp(-te.abs(a[b,h]))),name='softplus')
    decay=te.compute((batch,nv),lambda b,h:te.exp(-te.exp(al[h])*softplus[b,h]),name='decay')
    beta=te.compute((batch,nv),lambda b,h:1/(1+te.exp(-ba[b,h].astype('float32'))),name='beta')
    decayed=te.compute((batch,nv,dk,dv),lambda b,h,i,j:state[b,h,i,j]*decay[b,h],name='decayed_state')
    rd=te.reduce_axis((0,dk),'predict_k')
    prediction=te.compute((batch,nv,dv),lambda b,h,j:te.sum(decayed[b,h,rd,j]*kn[b,h//(nv//nk),rd],axis=rd),name='prediction')
    delta=te.compute((batch,nv,dv),lambda b,h,j:beta[b,h]*(conv_silu[b,2*kd+h*dv+j].astype('float32')-prediction[b,h,j]),name='delta')
    next_state=te.compute((batch,nv,dk,dv),lambda b,h,i,j:decayed[b,h,i,j]+kn[b,h//(nv//nk),i]*delta[b,h,j],name='NextState')
    rd=te.reduce_axis((0,dk),'out_k')
    attn=te.compute((batch,nv,dv),lambda b,h,j:te.sum(qn[b,h//(nv//nk),rd]*next_state[b,h,rd,j],axis=rd),name='attn')
    rd=te.reduce_axis((0,dv),'out_norm_d')
    ms=te.compute((batch,nv),lambda b,h:te.sum((attn[b,h,rd]*attn[b,h,rd]),axis=rd),name='gated_sum')
    gated=te.compute((batch,vd),lambda b,d:(attn[b,d//dv,d%dv]/te.sqrt(ms[b,d//dv]/dv+eps)*wg[d%dv]*qkvz[b,channels+d].astype('float32')/(1+te.exp(-qkvz[b,channels+d].astype('float32')))).astype('bfloat16'),name='gated_norm')
    proj=linear(gated,wo,hidden,vd,'out_proj')
    residual=te.compute((batch,hidden),lambda b,d:(x[b,d].astype('float32')+proj[b,d].astype('float32')).astype('bfloat16'),name='residual')
    post=rms(residual,wp,'post_norm')
    up=linear(post,wf,2*ff,hidden,'gate_up')
    act=te.compute((batch,ff),lambda b,d:(up[b,d].astype('float32')/(1+te.exp(-up[b,d].astype('float32')))*up[b,ff+d].astype('float32')).astype('bfloat16'),name='swiglu')
    down=linear(act,wd,hidden,ff,'down')
    output=te.compute((batch,hidden),lambda b,d:(residual[b,d].astype('float32')+down[b,d].astype('float32')).astype('bfloat16'),name='Output')
    next_conv=te.compute((batch,channels,3),lambda b,c,t:te.if_then_else(t<2,cs[b,c,t+1],qkvz[b,c]),name='NextConvState')
    return [x,wq,wb,wo,wf,wd,wc,wi,wp,wg,al,dt,cs,state,output,next_state,next_conv]
