import torch
from torch.utils.data import Dataset
import numpy as np
from torch_geometric.utils import to_undirected
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from ssm_utlis import set_seed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.notebook import tqdm
#set_seed(21)
from ssm_memory import MemoryModel

"""
Dataset creation,

for node $u = 0$, $ x_u \in \{-1,1\}$

else if  $u\neq 0$, $x_u = U[-1,1]$
"""

class CTDGraphDataset(Dataset):
    def __init__(self, N, M, seed=42):
        """
        Args:
            N (int): number of edges per sequence
            M (int): number of sequences
        """
        self.N = N
        self.M = M
        np.random.seed(seed)

        self.data = []
        self.labels = []

        for _ in range(M):
            # node 0 signal = label
            y = np.random.choice([0, 1])
            self.labels.append(y)

            # node signals
            signals = {0: float(y)*2-1}
            for i in range(1, N + 1):
                signals[i] = np.random.uniform(-1, 1)

            # edges
            src = np.arange(0, N)
            dst = np.arange(1, N + 1)
            times = np.sort(np.random.uniform(0, 10, N))  # increasing times

            # endpoint signals
            x_src = np.array([signals[s] for s in src])
            x_dst = np.array([signals[d] for d in dst])

            self.data.append((src, dst, times, x_src, x_dst))

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        src, dst, times, x_src, x_dst = self.data[idx]
        y = self.labels[idx]

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "dst": torch.tensor(dst, dtype=torch.long),
            "t": torch.tensor(times, dtype=torch.float32),
            "x_src": torch.tensor(x_src, dtype=torch.float32),
            "x_dst": torch.tensor(x_dst, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }

from torch.utils.data import DataLoader, random_split
def get_loaders(N, M, batch_size=4, train_split=0.7, val_split=0.15):
    """
    Args:
        N (int): number of nodes (dataset-specific param for CTDGraphDataset).
        M (int): number of samples.
        batch_size (int): batch size for loaders.
        train_split (float): fraction of data to use for training.
        val_split (float): fraction of data to use for validation.
        seed (int): random seed.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = CTDGraphDataset(N, M)

    # sizes
    train_size = int(train_split * M)
    val_size = int(val_split * M)
    test_size = M - (train_size + val_size)

    # make sure the total matches exactly
    assert train_size + val_size + test_size == M, "Split sizes must sum to dataset size"

    # split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator()
    )

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="best_model_sq.pt"):
        self.patience = patience
        self.delta = delta
        self.best_acc = -float("inf")
        self.counter = 0
        self.early_stop = False
        self.path = path  # save checkpoint

    def __call__(self, acc, model):
        if acc > self.best_acc + self.delta:
            self.best_acc = acc
            self.counter = 0
            # save model checkpoint
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model,train_loader,optimizer,criterion,delta,reg):
    model.train()
    total_loss, total_score, total_samples = 0, [], []
    pred = []
    hidden_state_list = []
    # yt = torch.zeros((len(train_loader),1),device=model.device)
    neighbour = 0

    for batch in train_loader:
        src, dst = batch["src"].to(device), batch["dst"].to(device)
        batch_size = src.shape[0]
        batch_ids = torch.arange(src.shape[0],device=device)
        x_src, x_dst, y = batch["x_src"].to(device), batch["x_dst"].to(device), batch["y"].to(device)
        src_b,dst_b = src+batch_ids[:,None],dst+batch_ids[:,None]
        m1_vec = torch.zeros((batch_size,model.num_nodes,model.hidden_dim),device=model.device)
        ne_vec = torch.zeros((batch_size,model.num_nodes,model.hidden_dim),device=model.device)
        # edge_index = torch.stack([torch.arange(batch_size*model.num_nodes,device=model.device),torch.arange(batch_size*model.num_nodes,device=model.device)])
        for t in range(src.shape[-1]):
            src_t = src_b[0,t]
            dst_t = dst_b[0,t]

            if t>0:
                neighbour = src[0,t-1]
            else :
                neighbour = src[0,t]

            src_tau = src[0,:t]
            dst_tau = dst[0,:t]

            L = torch.zeros((dst_t+1,dst_t+1),device=device)
            I = torch.eye(len(L),device=device)
            for u,v in zip(src_tau.tolist(),dst_tau.tolist()):
                L[u,v]=1
                L[v,u]=1

            D = torch.sum(L,dim=1)
            D[D==0]=1
            D = torch.diag(D**(-0.5))
            L_past = I - D@L.clone()@D

            L[src_t,dst_t]=1
            L[dst_t,src_t]=1

            D = torch.sum(L,dim=1)
            D[D==0]=1
            D = torch.diag(D**(-0.5))

            L_present = I - D@L.clone()@D

            active = torch.hstack([neighbour,src_t,dst_t])

            L_sub_present = L_present[active.min():active.max()+1,active.min():active.max()+1]
            L_sub_past = L_past[active.min():active.max()+1,active.min():active.max()+1]

            if t==0:
                L_sub_past = torch.eye(len(L_sub_past),device=device)

            In = torch.eye(len(L_sub_past )).to(device)
            x_t = torch.zeros((batch_size,len(L_sub_past ),model.input_dim),device=model.device)
            # x_n =  torch.zeros((batch_size,2,model.hidden_dim),device=model.device)

            # Fill Data to Current Node
            if t>0:
                x_t[:,-3] = x_src[:,t-1].unsqueeze(-1)


            x_t[:,-2]= x_src[:,t].unsqueeze(-1)
            x_t[:,-1]= x_dst[:,t].unsqueeze(-1)


            prev_c_1= m1_vec[:,torch.unique(active)].view((batch_size*(len(L_sub_past )),-1))
            x_batch = x_t.view(batch_size*len(L_sub_past ),-1)
            zt = model.TuneInputSC(x_batch) # input Tuning / Selective Scan

            A = model.A_
            At_ = torch.eye(len(A)).to(device) -A.T*delta #torch.matrix_exp(-A.T*delta)

            PL_ = model.w[0]*In + model.w[1]*L_sub_present  + model.w[2]*L_sub_present@L_sub_present
            PL_p = model.w[0]*In + model.w[1]*L_sub_past  + model.w[2]*L_sub_past@L_sub_past

            PL_inv = torch.linalg.inv(PL_)
            # A_s= 0*In # - PL_inv@PL_p # Uncomment to implement TU-SSM varaiant
            A_s= In  - PL_inv@PL_p
            PL_b = torch.block_diag(*PL_inv[None,:,:].repeat(batch_size,1,1))
            A_sb = torch.block_diag(*A_s.unsqueeze(0).repeat(batch_size,1,1))
            updated_c_1=  -A_sb@ prev_c_1 +  prev_c_1 @ At_  +  PL_b @ zt # simple linear approximation of CTT-HiPPO ODE using Forward Euler
            final_state = updated_c_1

            m1_vec[:,torch.unique(active)] = updated_c_1.view(batch_size,len(L_sub_past ),-1)
            ne_vec[:,torch.unique(active)] = final_state.view(batch_size,len(L_sub_past ),-1)

            # edge_index = current_edge_index.clone()

        hidden_state_list.append(ne_vec)
        logits = model.SeqClass(hidden_state_list[-1][:,-1]).view(-1)
        loss = criterion(target = y,input=logits) #+ torch.norm(model.A_hippo-model.A_,p=2)*1e-1
        total_loss += loss #/ batch_size
        # preds = ( logits> 0).float()
        pred += (logits>0).to(int).tolist()
        total_score += logits.tolist()
        total_samples += y.tolist()

        if torch.is_grad_enabled():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc = accuracy_score(pred,total_samples)
    if torch.is_grad_enabled():
        print(f"Epoch | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    else:
        print(f'val/test acc: {acc} ')

    return acc

v = []

N = 20           # No. of observation in a seq.
M=1000           # No. of Seq
batch_size = 400 # Batch Size for loading Seq in model, i.e., we process batch_size number of seq. in a parallel batch.
train_loader,val_laoder,test_loader = get_loaders(N=N,M=M,batch_size=batch_size)

for _ in range(10): # Running for 10 initilizations
    state_dim =32   # Memory Dim
    time_dim = 4    # Time Dim/ Not used
    device = 'cuda'
    lr = 1e-2
    epochs = 1600
    model = MemoryModel(num_nodes=N+1,input_dim=1,hidden_dim=state_dim,time_dim=time_dim,reg=1e-4,device=device,update_type='mamba').to(device)
    print(f'Init filter coff.: {model.w}')
    path = 'sq_mod_x.pt'
    early_stopper = EarlyStopping(patience=50, delta=1e-3,path=path)
    criterion = nn.BCEWithLogitsLoss()  # for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)

    delta = 1/(N+1) # Using non-learnable delta
    for epoch in tqdm(range(epochs),desc='Epochs: '):
        train_model(model,train_loader,optimizer,criterion,delta,0.1)

        with torch.no_grad():
            val_acc = train_model(model,val_laoder,optimizer,criterion,delta,0.1)
        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(path,weights_only=True))
    print(f'Trained filter coff. :{model.w}')
    with torch.no_grad():
        test_acc = train_model(model,test_loader,optimizer,criterion,delta,0.1)
        v.append(test_acc)

print(f"Mean: {np.mean(v)}, std:{np.std(v)}")