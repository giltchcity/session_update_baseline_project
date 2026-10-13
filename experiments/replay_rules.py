# Offline replay of the element rule on a session's rows and real/synthetic frames (1 Hz), several evidence models, GT/reference only labels the outcome.
#   usage: replay_rules.py <realB|realC|synB> <run> <rows_per_stratum>
# Rows: alive at the session end + rows ended by evidence (positions of the checkpoint). Strata: carried (created before the session) x object/background.
# Variants:  K0 as coded, scope background + carried-object rows (0c9c487 scope);  K1 as coded, all rows (F2/F4 scope);
#            Q2 per-round CUSUM, step = -ln q(range), q = through share of the rows with an earlier hit in this session at that range, learned online (all rows);
#            Q2b same, scope of K0;  Q1 per-stamp CUSUM with the same q(range) (all rows).
import sys, csv, math, numpy as np, torch, tifffile, open3d as o3d
from scipy.spatial import cKDTree
R="/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"; G="/home/jixian/Desktop/FT/results/abc_eval_v2/geometry"; D="/home/jixian/Desktop/FT/datasets/synthetic_ab"
MAX=torch.iinfo(torch.int64).max; LN99=math.log(99.0); TOL=0.05
which,run,NS=sys.argv[1],sys.argv[2],int(sys.argv[3])
sess={"realA":"a","realB":"b","realC":"c","synB":"b"}[which]
ck=torch.load(f"{R}/{run}/checkpoint_{sess}.pt",map_location="cpu",weights_only=False,mmap=True); b=ck["backend"]; n=len(b["identity"]); m=b["model"]
xyz_all=(m[1].detach().float()/10.0).numpy(); ext_all=(3.0*torch.exp(m[4].detach().float()).max(dim=1).values/10.0).numpy()
ident=b["identity"].numpy(); de=b["death_evidence"].numpy()<MAX; ds=b["death_state"].numpy()<MAX; dp=b["death_prune"].numpy()<MAX
LUall=b["last_update"].numpy(); CRall=b["created"].numpy(); st=int(b["session_starts"][-1]); tracked=set(b["tracked_labels"].tolist()); obj_all=np.isin(ident,list(tracked)); carried_all=CRall<st
cand=np.nonzero(~(ds|dp)|de)[0]       # alive rows + rows ended by evidence (not rows ended by state closure or prune)
rng=np.random.default_rng(5); sel=[]; strat=[]
for c in (0,1):
    for o in (0,1):
        idx=cand[(carried_all[cand]==bool(c))&(obj_all[cand]==bool(o))]; k=min(NS,len(idx)); sel.append(np.sort(rng.choice(idx,size=k,replace=False)) if k else np.zeros(0,dtype=np.int64)); strat.append((c,o,len(idx),k))
sel=np.concatenate(sel); P=xyz_all[sel].astype(np.float64); E=ext_all[sel].astype(np.float64); LU=LUall[sel]; CR=CRall[sel]; RUN_END=de[sel]; car=carried_all[sel]; obj=obj_all[sel]; N=len(P)
w_row=np.concatenate([np.full(k,tot/max(k,1)) for (_,_,tot,k) in strat])   # scale a sampled row to the population
if which=="synB":
    g=np.load(f"{D}/session_b/gt/temporal_gt.npz"); ids=list(g["instance_ids"]); z=np.load(f"{D}/gt_geometry/b_local_meshes.npz"); nb=len(g["render_visible"]); pts=[]
    for i in ids:
        j=ids.index(i)
        if not g["render_visible"][nb-1,j]: continue
        T=g["T_world_object"][nb-1,j]; V=z[f"{i:04d}_vertices"].astype(np.float64); F=z[f"{i:04d}_faces"].astype(np.int32); V=(np.c_[V,np.ones(len(V))]@T.T)[:,:3]
        mm=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F)); pts.append(np.asarray(mm.sample_points_uniformly(number_of_points=max(100,int(mm.get_surface_area()*4000))).points))
    dref=cKDTree(np.vstack(pts)).query(P,k=1,distance_upper_bound=2.0)[0]
    from scipy.spatial.transform import Rotation as Rot
    tum=np.loadtxt(f"{D}/session_b/trajectory_gt.tum"); Rq=Rot.from_quat(tum[:,4:8]).as_matrix(); tw_=tum[:,1:4]; fx=fy=588.897289890142; cx,cy,W_,H_=340.0,240.0,680,480
    fr=[(st+int(f/30*1e9),Rq[f],tw_[f],f"{D}/session_b/rgbd/{f:06d}_depth.tiff") for f in range(0,len(tum),30)]; ROUND=10.8e9
else:
    root="/home/jixian/Desktop/FT/datasets/local_ab"; tag={"realA":"a_20260809_204010_201","realB":"b_20260810_030502_620","realC":"c_20260822_182703_387"}[which]; rb=f"{root}/rgbd/session_{tag}_flat_rgbd_30hz_1080p"
    Tba=np.eye(4) if which=="realA" else np.loadtxt(f"{root}/alignment/session_{sess}_to_session_a.txt"); ref=np.asarray(o3d.io.read_triangle_mesh(f"{G}/{sess.upper()}_reference_1cm.ply").vertices,dtype=np.float64); dref=cKDTree(ref).query(P,k=1,distance_upper_bound=2.0)[0]
    ts=[(int(r[0]),int(r[1])) for r in list(csv.reader(open(f"{rb}/timestamps.csv")))[1:] if r and r[0].isdigit()]; fr=[]; last=None
    for fid,stamp in ts:
        if last is not None and stamp-last<int(1e9): continue
        last=stamp; T=Tba@np.loadtxt(f"{rb}/{fid:06d}_pose.txt"); fr.append((st+stamp,T[:3,:3],T[:3,3],f"{rb}/{fid:06d}_depth.tiff"))
    fx,fy,cx,cy,W_,H_=958.539672852,958.00012207,956.520463638,548.005616755,1920,1080; ROUND=2.1e9
onref=dref<=0.05
n_,s_=[float(x) for x in open(f"{R}/{run}/session_{sess}/sensor_statistics.txt").read().strip().splitlines()[-1].split()[:2]]; p_miss=s_/n_; step0=-math.log(min(0.995,max(0.005,p_miss)))
scope0=(ident[sel]<=0)|(car&obj)      # 0c9c487 scope: background + memory object rows (approx of [M1])
edges=np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,5,6,8,99]); NB=len(edges)-1
def st_(): return dict(c=np.zeros(N),vl=np.zeros(N,np.int8),ret=np.zeros(N,bool))
K0,K1,Q2,Q2b=st_(),st_(),st_(),st_(); cQ1=np.zeros(N); retQ1=np.zeros(N,bool)
cntOn=np.zeros(NB); cntThr=np.zeros(NB); hit_before=np.zeros(N,bool); round_end=None
Trc=np.zeros((N,NB),np.int32); Nrc=np.zeros((N,NB),np.int32); cH=np.zeros(N); retH=np.zeros(N,bool); popinfo=[]
def beta_pop():
    # method of moments per range bin over the rows with an earlier hit and >= 2 verdicts in the bin: mean mu and intra-row correlation phi -> (a, b)
    ab=np.zeros((NB,2)); 
    for r in range(NB):
        m_=hit_before&(Nrc[:,r]>=2)
        if m_.sum()<30: ab[r]=(0.05*1000,0.95*1000); continue          # too few rows: independent Bernoulli at the pooled share (as coded)
        t=Trc[m_,r].astype(float); nn=Nrc[m_,r].astype(float); mu=t.sum()/nn.sum(); mu=min(max(mu,1e-3),1-1e-3); pi=t/nn
        V=(nn*(pi-mu)**2).sum()/nn.sum(); Vb=mu*(1-mu)*m_.sum()/nn.sum(); phi=(V-Vb)/max(mu*(1-mu)-Vb,1e-9); phi=min(max(phi,1e-3),0.999); kappa=1.0/phi-1.0
        ab[r]=(mu*kappa,(1-mu)*kappa)
    return ab
for stamp_abs,Rw,tw,dpath in fr:
    dep=np.nan_to_num(tifffile.imread(dpath).astype(np.float64),nan=0.0); stamp=stamp_abs-st
    pc=(P-tw)@Rw; zz=pc[:,2]; ok=zz>0.1; u=np.round(fx*pc[:,0]/np.where(ok,zz,1)+cx).astype(int); v=np.round(fy*pc[:,1]/np.where(ok,zz,1)+cy).astype(int)
    inv=ok&(u>=0)&(u<W_)&(v>=0)&(v<H_); dm=np.zeros(N); dm[inv]=dep[v[inv],u[inv]]; meas=inv&(dm>0); band=TOL+E; d=dm-zz; later=stamp_abs>LU
    on=meas&(np.abs(d)<=band)&later; thr=meas&(d>band)&later
    bi=np.clip(np.searchsorted(edges,zz,side="right")-1,0,NB-1); tot=cntOn+cntThr; q_all=(cntThr.sum()/tot.sum()) if tot.sum()>=3 else 0.05
    q=np.clip(np.where(tot>=20,(cntThr+1.0)/(tot+2.0),q_all),0.005,0.995); stepz=-np.log(q)[bi]
    for S,stp in ((K0,None),(K1,None),(Q2,stepz),(Q2b,stepz)): S["vl"][on&~S["ret"]]=1; S["vl"][thr&~S["ret"]]=2; S["stp"]=stp
    cQ1[on&~retQ1]=0; cQ1[thr&~retQ1]+=stepz[thr&~retQ1]; retQ1|=thr&(cQ1>LN99)
    ab=beta_pop(); a_=ab[bi,0]; b_=ab[bi,1]; tI=Trc[np.arange(N),bi]; nI=Nrc[np.arange(N),bi]
    pred=np.clip((a_+tI)/(a_+b_+nI),0.005,0.995); ell=-np.log(pred)
    cH[on&~retH]=0; cH[thr&~retH]+=ell[thr&~retH]; retH|=thr&(cH>LN99)
    act=(on|thr)       # the population keeps every verdict of every row (retired rows included: no survivor bias)
    np.add.at(Nrc,(np.nonzero(act)[0],bi[act]),1); np.add.at(Trc,(np.nonzero(thr)[0],bi[thr]),1)
    cntOn+=np.bincount(bi[hit_before&on],minlength=NB); cntThr+=np.bincount(bi[hit_before&thr],minlength=NB); hit_before|=on
    if round_end is None: round_end=stamp+ROUND
    if stamp>=round_end:
        for S in (K0,K1,Q2,Q2b):
            a=(S["vl"]==1)&~S["ret"]; S["c"][a]=0; t=(S["vl"]==2)&~S["ret"]; S["c"][t]+=(step0 if S["stp"] is None else S["stp"][t]); S["ret"]|=t&(S["c"]>LN99); S["vl"][:]=0
        round_end=stamp+ROUND
variants={"run (checkpoint evidence ends)":RUN_END,"K0 coded, 0c9c487 scope":K0["ret"]&scope0,"K1 coded, all rows (F2/F4)":K1["ret"],"Q2 per-round, q(range) learned, all rows":Q2["ret"],"Q2b per-round, q(range), 0c9c487 scope":Q2b["ret"]&scope0,"Q1 per-stamp, q(range), all rows":retQ1,"H1 per-stamp, beta-binomial per range, all rows":retH}
print(f"\n{which} {run}: {N} sampled rows of {sum(t for *_,t,_ in strat)}; strata (carried,obj): "+" ".join(f"({c},{o}):{k}/{tot}" for c,o,tot,k in strat)+f"; pooled p_miss {p_miss:.4f} step {step0:.2f}")
for r_ in (2,5,6,7):
    m_=hit_before&(Nrc[:,r_]>=2); print(f"   debug bin {r_}: rows with hit_before & n>=2: {m_.sum()}, sum t {Trc[m_,r_].sum()}, sum n {Nrc[m_,r_].sum()}, all rows sum t {Trc[:,r_].sum()} sum n {Nrc[:,r_].sum()}")
ab=beta_pop(); print("H1 population at the end (range bin: mean q, a+b): "+" ".join(f"{edges[k]:.1f}:{ab[k,0]/(ab[k,0]+ab[k,1]):.2f},{ab[k,0]+ab[k,1]:.1f}" for k in range(NB)))
print("learned q(range) at the end: "+" ".join(f"{edges[k]:.1f}:{(cntThr[k]+1)/(cntOn[k]+cntThr[k]+2):.2f}(n={cntOn[k]+cntThr[k]:.0f})" for k in range(NB)))
print(f"{'variant':44s} | total retired (est) | on-ref% | carried: n on-ref% | this-session: n on-ref% | object rows: n | bg rows: n")
for nm,ret in variants.items():
    tot_=(ret*w_row).sum(); fp=((ret&onref)*w_row).sum()/max(tot_,1)*100
    def part(mask): 
        t=(ret&mask)*w_row; return t.sum(), ((ret&onref&mask)*w_row).sum()/max(t.sum(),1)*100
    c1=part(car); c2=part(~car); o1=part(obj); o2=part(~obj)
    print(f"{nm:44s} | {tot_:9.0f} | {fp:5.1f} | {c1[0]:8.0f} {c1[1]:5.1f} | {c2[0]:8.0f} {c2[1]:5.1f} | {o1[0]:8.0f} | {o2[0]:8.0f}")
if which=="synB":
    pl=(ident[sel]==109); print(f"plant 109 rows in sample {pl.sum()}: H1 {(retH&pl).sum()}, retired K0 {(K0['ret']&scope0&pl).sum()} K1 {(K1['ret']&pl).sum()} Q2 {(Q2['ret']&pl).sum()} Q1 {(retQ1&pl).sum()}; off-GT>10cm among them: {(dref[pl]>0.10).sum()}")
    off=(dref>0.10)
    for nm,ret in variants.items(): print(f"   {nm:44s} retired rows off the GT > 10 cm: {((ret&off)*w_row).sum():9.0f}  of all retired {(ret*w_row).sum():9.0f}")

# by identity (sampled rows scaled to the population): who is retired
idn=ident[sel]; print("\nretired rows by identity (est. rows | share on the reference) -- K0 coded scope vs H1 (beta-binomial per stamp), all rows")
cands=sorted(set(idn.tolist()), key=lambda i:-(((K0["ret"]&scope0)&(idn==i))*w_row).sum())[:10]
for i in cands:
    mk=idn==i; r0=(K0["ret"]&scope0)&mk; r2=retH&mk; rs=RUN_END&mk
    print(f"   identity {i:4d}: sample rows {mk.sum():6d} (est {w_row[mk].sum():8.0f}); run-ended {(rs*w_row).sum():7.0f} ({(((rs&onref)*w_row).sum()/max((rs*w_row).sum(),1))*100:4.0f} %); K0 {(r0*w_row).sum():7.0f} ({(((r0&onref)*w_row).sum()/max((r0*w_row).sum(),1))*100:4.0f} %); H1 {(r2*w_row).sum():7.0f} ({(((r2&onref)*w_row).sum()/max((r2*w_row).sum(),1))*100:4.0f} %)")

import json
print("RESULT "+json.dumps({nm:dict(ended=float((ret*w_row).sum()),false=float(((ret&onref)*w_row).sum())) for nm,ret in variants.items()}))
