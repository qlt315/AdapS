import os, time, warnings, torch, torch.nn.functional as F
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model          # <— 改：统一引入
from utils.modulation  import QAM, PSK
from utils.accuracy    import accuracy

# ---------- utils --------------------------------------------------------- #
def infer_base_and_num_classes(dataset_name: str):
    name = dataset_name.strip().upper()
    if name.startswith("CIFAR100"):
        return "CIFAR100", 100
    if name.startswith("CINIC10"):
        return "CINIC10", 10
    if name.startswith("CIFAR10"):
        return "CIFAR10", 10
    raise ValueError(f"Unsupported dataset: {dataset_name}")

def build_backbone(base: str, num_classes: int, args):
    if base == "CIFAR10":
        return JSCC_model.DTJSCC_CIFAR10(3, args.lat_d, num_classes, num_embeddings=args.num_emb)
    if base == "CIFAR100":
        return JSCC_model.DTJSCC_CIFAR100(3, args.lat_d, num_classes, num_embeddings=args.num_emb)
    if base == "CINIC10":
        return JSCC_model.DTJSCC_CINIC10(3, args.lat_d, num_classes, num_embeddings=args.num_emb)
    raise ValueError(f"No model available for base={base}")

def build_stream(base: str):
    return [
        ("awgn",    base),
        ("rician",  base),
        ("rayleigh",base),
        ("awgn",    f"{base}_noise"),
        ("rician",  f"{base}_noise"),
        ("rayleigh",f"{base}_noise"),
    ]

# ---------- single domain (support-adapt + query-eval) -------- #
def run_one_domain(model, mod, ds_name, chan, args, domain_id):
    dev=args.device
    # ------------ support (inner-loop) ------------------------- #
    if args.adapt and args.adapt_steps>0:
        sup_loader = get_data(ds_name,args.inner_bs,n_worker=0,train=False)
        sup_iter   = iter(sup_loader)
        opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr, momentum=0.9)

        t0=time.time(); model.train()
        for _ in range(args.adapt_steps):
            try:
                x,y = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_loader)
                x,y = next(sup_iter)
            x,y = x.to(dev),y.to(dev)
            opt.zero_grad()
            out,_ = model(x,mod=mod)
            F.cross_entropy(out,y).backward(); opt.step()
        adapt_ms=(time.time()-t0)*1e3
        model.eval()
    else:
        adapt_ms=0.0

    # ------------ query --------------------------------------- #
    qry_loader = get_data(ds_name,256,n_worker=4,train=False)
    c1=c3=s=0
    with torch.no_grad():
        for x,y in qry_loader:
            x,y = x.to(dev),y.to(dev)
            out,_ = model(x,mod=mod)
            t1,t3 = accuracy(out,y,(1,3)); bs=y.size(0)
            c1+=t1.item()*bs/100; c3+=t3.item()*bs/100; s+=bs
    acc1,acc3 = c1/s*100, c3/s*100
    print(f"[{domain_id}] {chan.upper():>8} | {ds_name:<15} "
          f"Acc1 {acc1:6.2f}  Acc3 {acc3:6.2f}  Adapt {adapt_ms:6.1f} ms")
    return acc1,acc3,adapt_ms

# ---------- main stream -------------------------------------- #
def main(args):
    # ---- choose base & classes -------------------------------- #
    base, num_classes = infer_base_and_num_classes(args.dataset)

    # ---- build backbone -------------------------------------- #
    net = build_backbone(base, num_classes, args)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=FutureWarning)
        ckpt=torch.load(args.model_path,map_location='cpu',weights_only=False)
    net.load_state_dict(ckpt["model_states"] if "model_states" in ckpt else ckpt)
    net.to(args.device)

    # ---- 6-domain stream ------------------------------------- #
    stream = build_stream(base)

    SNR=0
    acc1s, acc3s, adapt_ms = [], [], []
    for idx,(chan,ds) in enumerate(stream,1):
        # channel object --------------------------------------- #
        kw={"K":args.K} if chan=="rician" else {}
        if chan=="nakagami": kw["m"]=args.m
        mod=(QAM if args.mod=='qam' else PSK)(args.num_emb,SNR,chan,kw)

        # run one domain -------------------------------------- #
        a1, a3, t = run_one_domain(net, mod, ds, chan, args, idx)
        acc1s.append(a1); acc3s.append(a3); adapt_ms.append(t)

        # save snapshot for next domain ------------------------ #
        os.makedirs(args.ckpt_dir,exist_ok=True)
        torch.save(net.state_dict(),
                   os.path.join(args.ckpt_dir,f"meta_after_domain{idx}.pt"))

    # ---- dump results ---------------------------------------- #
    os.makedirs(args.save_dir,exist_ok=True)
    torch.save({"acc1":acc1s,"acc3":acc3s,"adapt_ms":adapt_ms},
               os.path.join(args.save_dir,"meta_stream_results.pt"))
    print("\n[Done] Results written to meta_stream_results.pt")

# ---------- CLI ---------------------------------------------- #
if __name__=="__main__":
    import argparse
    P=argparse.ArgumentParser("Stream Meta-eval")
    # model / data
    P.add_argument("--dataset",default="CIFAR10")
    P.add_argument("--num_emb",type=int,default=16)
    P.add_argument("--lat_d",type=int,default=512)
    P.add_argument("--num_lat",type=int,default=4)
    # channel & mod
    P.add_argument("--mod",choices=["qam","psk"],default="psk")
    P.add_argument("--K",type=float,default=3.0)
    P.add_argument("--m",type=float,default=2.0)
    # adapt
    P.add_argument("--adapt",action="store_true",default=True)
    P.add_argument("--adapt_steps",type=int,default=5)
    P.add_argument("--inner_bs",type=int,default=256)
    P.add_argument("--inner_lr",type=float,default=1e-2)
    # paths
    P.add_argument("--model_dir",default="Meta-JSCC/trained_models")
    P.add_argument("--model_file",default="meta_best.pt")
    P.add_argument("--save_dir",default="Meta-JSCC/eval")
    P.add_argument("--ckpt_dir",default="Meta-JSCC/ckpt_stream")
    # runtime
    P.add_argument("--device",default="cuda:0")
    args=P.parse_args()
    args.device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.model_path=os.path.join(args.model_dir,args.model_file)
    main(args)
