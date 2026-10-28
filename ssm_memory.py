import torch
import torch.nn as nn
import math
from torch_geometric.nn import GATConv
from ssm_dataset import active_data
import torch.nn.functional as F
from ssm_utlis import scatter_mean_update

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))  # learnable γ

    def forward(self, x):
        # x: (batch, seq_len, d_model) or (batch, d_model)
        norm = x.norm(dim=-1, keepdim=True)  # L2 norm across features
        rms = norm / (x.size(-1) ** 0.5)     # divide by sqrt(d)
        return self.scale * (x / (rms + self.eps))

def init_hippo_matrices( D, input_dim):
        A = torch.zeros(D, D)
        B = torch.zeros(D, input_dim)
        for i in range(D):
            for j in range(D):
                if i > j:
                    A[i,j]=math.sqrt(2*i + 1) * math.sqrt(2*j + 1)*((-1)**(i-j))
                else:
                    A[i,j]=math.sqrt(2*i + 1) * math.sqrt(2*j + 1)
                    #A[i, j] = math.sqrt(2*i + 1) * math.sqrt(2*j + 1)
                #elif i == j:
                    #A[i, j] = i + 1
            for k in range(input_dim):
                B[i, k] = math.sqrt(2*i + 1)

        V = torch.linalg.eigvals(A)
        
        A /= V.abs().max()  # normalize for stability by eigne values

        return A, B



class TimeEncode(nn.Module):
    """
    TGAT-style learnable harmonic time encoding:
      phi(t) = cos(W * t + b),   W,b learnable (shape: [time_dim])
    Inputs can be any shape [..., 1]; output is [..., time_dim].
    """
    def __init__(self, time_dim: int = 16, init_scale: float = 1.0):
        super().__init__()
        self.time_dim = time_dim
        # Initialize frequencies on a log scale (stable start)
        freqs = torch.logspace(-4, 4, steps=time_dim) * init_scale
        self.W = nn.Parameter(freqs)          # [time_dim]
        self.b = nn.Parameter(torch.zeros(time_dim))  # [time_dim]

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        dt: [..., 1] time difference (e.g., current_ts - neighbor_ts), in seconds
            Recommended: normalize (e.g., divide by 3600 or 86400).
        returns: [..., time_dim]
        """
        # Ensure last dim is 1
        if dt.dim() == 0:
            dt = dt.view(1, 1)
        if dt.size(-1) != 1:
            dt = dt.unsqueeze(-1)
        # Broadcast multiply: [..., 1] * [time_dim] -> [..., time_dim]
        phase = dt * self.W + self.b
        return torch.cos(phase)

class StateProjection(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim:int = 16):
        super().__init__()
        self.w = nn.Linear(input_dim,output_dim)

    def forward(self,h_t,delt_t):

        return h_t + delt_t[:,None]*self.w(h_t)
    

class Time2Vec(nn.Module):
    """
    Time2Vec layer from Shukla et al. 2020.
    
    Input:  Δt tensor of shape [N] or [N,1]
    Output: time embeddings of shape [N, k+1]
    - First dimension: linear term
    - Remaining k dimensions: periodic terms
    """
    def __init__(self, k: int, activation=torch.sin):
        super().__init__()
        self.k = k
        self.activation = activation

        # Linear term (ω0, b0)
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))

        # Periodic terms (ωi, bi) for i=1..k
        self.W = nn.Parameter(torch.randn(k))
        self.B = nn.Parameter(torch.randn(k))

    def forward(self, delta_t: torch.Tensor):
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)  # [N] -> [N,1]

        # Linear term
        linear = self.w0 * delta_t + self.b0  # [N,1]

        # Periodic terms
        periodic = self.activation(delta_t * self.W + self.B)  # [N,k]

        return torch.cat([linear, periodic], dim=1)  # [N, k+1]
    
    
class static_embedding(torch.nn.Module):
    def __init__(self,out_dims,num_nodes,typ='LR',device='cuda'):
        super().__init__()
        self.out_dims = out_dims
        self.num_nodes = num_nodes
        self.typ = typ
        if typ=='LR':
            self.encoder = nn.Embedding(num_embeddings=self.num_nodes,embedding_dim=self.out_dims)
            self.dims = self.out_dims

        elif typ=='OH':
            self.encoder = torch.eye(num_nodes).to(device)
            self.dims = num_nodes

        elif typ=='RN':
            # self.encoder = nn.Embedding(num_embeddings=self.num_nodes,embedding_dim=self.out_dims,_freeze = True).to(device)
            self.encoder = torch.empty((num_nodes,out_dims),dtype=torch.float32,device=device)
            self.dims = self.out_dims

        elif typ=='Nil':
            self.encoder =  torch.zeros(num_nodes,1).to(device)
            self.dims = 1

        else:
            print('No valid type selected')
            raise TypeError
        
    def forward(self,idx):
        if self.typ=='OH' or self.typ =='Nil' or self.typ=='RN':
            return self.encoder[idx]
        else:
            return self.encoder(idx)


class MemoryModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim,time_dim,reg, delta=0.1,embd_dims=32 ,device='cuda',update_type ='mamba',use_old_message=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.delta = delta
        self.bound = 0
        self.time_dim = time_dim
        self.input_dim = input_dim

        self.update_type = update_type
        # if self.update_type == 'HiPPO':
        #     A,B_h = init_hippo_matrices(hidden_dim,1)
        #     self.A_hippo = nn.Parameter(A.to(device),requires_grad=False)
        #     self.B_hippo = nn.Parameter(B_h.to(device),requires_grad=True)
        # else :

        self.A_log_1 = nn.Parameter(torch.randn(self.hidden_dim))
        self.A_log_2 = nn.Parameter(torch.randn(self.hidden_dim))

        self.device = device
        self.rms_1 = RMSNorm(hidden_dim)
        self.rms_2 = RMSNorm(hidden_dim)
        # self.dropout = nn.Dropout(dropout) 
        self.reg = nn.Parameter(torch.tensor(reg),requires_grad=False)
        self.B1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.B2 = nn.Linear(self.hidden_dim,self.hidden_dim)

        # Stable Initilizations
        # nn.init.kaiming_normal_(self.B1.weight,mode='fan_in',nonlinearity='relu')
        # nn.init.kaiming_normal_(self.B2.weight,mode='fan_in',nonlinearity='relu')
        # if self.B1.bias is not None:
        #     nn.init.zeros_(self.B1.bias)
        #     nn.init.zeros_(self.B2.bias)

        self.project_t = StateProjection(input_dim=self.hidden_dim,output_dim=self.hidden_dim)
        A,B_h = init_hippo_matrices(hidden_dim,self.input_dim)
        self.w = nn.Parameter(torch.tensor([0.25,10.0,20.0]).to(device),requires_grad=True)
        # self.A_hippo = nn.Parameter(A.to(device),requires_grad=False)
        # self.B_hippo = nn.Parameter(B_h.to(device),requires_grad=True)
        self.A_ = nn.Parameter(A.to(device),requires_grad=True)
        self.B_ = nn.Parameter(B_h.to(device),requires_grad=False)
        # self.embd_dims = embd_dims
        self.dt_proj_a = nn.Linear(hidden_dim, 1)
        self.dt_proj_b = nn.Linear(hidden_dim, 1)
        typ = 'Nil'
        self.names = static_embedding(embd_dims,self.num_nodes,typ=typ,device=self.device)#F.one_hot(torch.arange(self.num_nodes),num_classes=self.num_nodes)#torch.eye(self.num_nodes)#nn.Embedding(num_embeddings=num_nodes,embedding_dim=self.embd_dims,_freeze=True)
        print(f'Static Embeddings: {typ}')
        self.embd_dims = self.names.dims

        self._init_dt_bias(1e-3, 1e-1)

        self.NodeClassificaiton = nn.Sequential(
            nn.Linear(self.hidden_dim+self.time_dim+1, 1),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim,1),
        )

        self.PredictionMap = nn.Sequential(
            nn.Linear(2*self.embd_dims+2*self.hidden_dim+self.time_dim+1, 1),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim,1),
        )
        
        self.TuneInput = nn.Sequential(
            nn.Linear(input_dim+2*self.embd_dims, hidden_dim),#nn.Linear(input_dim+2*self.hidden_dim, hidden_dim),#nn.Linear(input_dim+2*self.embd_dims, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim,hidden_dim)
  
        )

        
        self.TuneInputSC = nn.Sequential(
            # nn.Linear(input_dim,hidden_dim)
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.input_dim)
        )

        self.SeqClass = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            # nn.ReLU(),
            # nn.Linear(64 ,1),
        )
        
        # self.TuneInput.apply(lambda m: self.init_weight(m))
        # self.TuneInputSC.apply(lambda m: self.init_weight(m))
        # self.PredictionMap.apply(lambda m: self.init_xav(m))

        # self.TuneInputNC = nn.Sequential(
        #     nn.Linear(input_dim+self.embd_dims, hidden_dim),
        #     nn.ReLU(),
        #      nn.Linear(self.hidden_dim,hidden_dim),
        # )

        # self.names = nn.Embedding(num_embeddings=num_nodes,embedding_dim=time_dim)
        self.time_encoder = Time2Vec(self.time_dim)
        # self.init_c = nn.Parameter(torch.randn(num_nodes, hidden_dim),requires_grad=False)
        # self.m1_dict = dict()
        # self.m2_dict = dict()
        # self.ne_dict = dict()
        self.m1_vec = torch.zeros(num_nodes, hidden_dim, device=device)
        self.m2_vec =  torch.zeros(num_nodes, hidden_dim, device=device)
        self.ne_vec = torch.zeros(num_nodes, hidden_dim, device=device)
        self.retain_message = use_old_message
        if use_old_message:
            self.old_message = torch.zeros((self.num_nodes,input_dim+2*self.embd_dims),device=device)


        # Nodes of 8-point Quad

        nodes = torch.tensor([
        -0.1834346424956498,
        -0.5255324099163290,
        -0.7966664774136267,
        -0.9602898564975363,
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
        ], dtype=torch.float32).to(device)

        weights = torch.tensor([
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ], dtype=torch.float32).to(device)

        tau = 1.0  # change as needed
        self.t_nodes = 0.5 * tau * (nodes + 1.0)
        self.t_weights = 0.5 * tau * weights

        
    def init_weight(self,module,activation='relu'):
        if isinstance(module,nn.Linear):
            nn.init.kaiming_uniform_(module.weight,nonlinearity = activation)
            nn.init.zeros_(module.bias)
    
    def init_xav(self,module):
        if isinstance(module,nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def reset(self):
        self.m1_vec = torch.zeros(self.num_nodes, self.hidden_dim, device=self.device)
        self.m2_vec =  torch.zeros(self.num_nodes, self.hidden_dim, device=self.device)
        self.ne_vec = torch.zeros(self.num_nodes, self.hidden_dim, device=self.device)
        if self.retain_message:
            self.old_message = torch.zeros((self.num_nodes,self.input_dim+2*self.embd_dims),device=self.device)
        
        # self.m1_dict = dict()
        # self.m2_dict = dict()
        # self.ne_dict = dict()

    def detach(self):
        self.m1_vec = self.m1_vec.detach()
        self.m2_vec =  self.m2_vec.detach() 
        self.ne_vec = self.ne_vec.detach()
        if self.retain_message:
            self.old_message = self.old_message.detach()

    def dict_lookup(self,keys_tensor, dictionary,mem_size):
         
        # Infer the vector size from the first dictionary value
        vector_size = mem_size
        results = torch.zeros((len(keys_tensor),vector_size)).to(self.device)
        
        for i,k in enumerate(keys_tensor):
            k = k.item()
            if k not in dictionary:
                dictionary[k] = torch.zeros(vector_size).to(torch.float32).to(self.device) 
            results[i] = dictionary[k]

        return results

    def _init_dt_bias(self, low, high):
        """Initialize dt bias so softplus(bias) ~ U[low, high]"""
        with torch.no_grad():
            # inverse softplus
            inv_sp = lambda y: y + torch.log(-torch.expm1(-y))
            self.dt_proj_a.bias.copy_(inv_sp(torch.empty_like(self.dt_proj_a.bias).uniform_(low, high)))
            self.dt_proj_a.weight.zero_()  # no weight influence at init
            self.dt_proj_b.bias.copy_(inv_sp(torch.empty_like(self.dt_proj_b.bias).uniform_(low, high)))
            self.dt_proj_b.weight.zero_()  # no weight influence at init

    def zoh_update(self,data:active_data):
        L_t = data.L_agg
        Is = torch.eye(len(L_t), device=L_t.device)
        delta_L = data.delta_L_agg
        As_bar = Is - self.reg*delta_L + self.reg**2 * delta_L @ (L_t + delta_L/2)
        
        dt = delta_L[None] * self.t_nodes[:,None,None]
        #dt2 = dt@dt # (8,n,n)
        LHS = Is[None] - self.reg*dt + self.reg**2*(dt@(L_t[None] + dt/2)) #- reg**3 * dt @ (L_t2[None] + L_t[None]@dt + dt2/6) 
        P_L_inv = Is - self.reg*L_t #+ reg**2 * L_t2 ( Linear Approximation )

        # Add Contextual Info to Signal

        neigh_info = torch.zeros((len(data.x_sub),2*self.embd_dims),device=self.device)
        neigh_info[:data.target_src.numel(),:self.embd_dims] = self.names(data.target_src)
        # neigh_info[:data.target_src.numel(),2*self.embd_dims:] = postive_time_enc
        neigh_info[:data.target_src.numel(),self.embd_dims:2*self.embd_dims] = self.names(data.target_dst)

        neigh_info[data.target_src.numel():2*data.target_src.numel(),:self.embd_dims] = self.names(data.target_dst)
        neigh_info[data.target_src.numel():2*data.target_src.numel(),self.embd_dims:2*self.embd_dims] = self.names(data.target_src)
        # neigh_info[data.target_src.numel():2*data.target_src.numel(),2*self.embd_dims:] = postive_time_enc
        # print(neigh_info.shape,data.x_sub.shape)
        x_in = torch.cat([data.x_sub,neigh_info],dim=1)

        if self.retain_message: # Use old message for neighbours nodes
            x_in[2*data.target_src.numel():] = self.old_message[data.active_input_ids[2*data.target_src.numel():]] #.detach()

        zt = self.TuneInput(x_in) # input Tuning / Selective Scan
        ztn = self.rms_1(zt)
        Bzt = self.B1(ztn)
        delta_1 =  F.softplus(self.dt_proj_a(ztn).expand(-1, self.hidden_dim)) # (n x d) delta for each input
        A_cont_1 = -torch.exp(self.A_log_1) # ( 1 x d)
        At_bar_1 = torch.exp(delta_1 * A_cont_1) # ( n x d)

        C_1 = (P_L_inv @ (Bzt * delta_1))[None]*self.t_weights[:,None,None] 
        RHS_1 = torch.exp((delta_1 * A_cont_1)[None] * self.t_nodes[:,None,None] )
        integral_1 = torch.sum(LHS@C_1*RHS_1,dim=0)

        updated_c_1= As_bar@(self.m1_vec[data.active_input_ids] * At_bar_1) + integral_1
        u_2 = zt+F.gelu(updated_c_1,approximate='tanh') #ztn #F.gelu(updated_c_1,approximate='tanh')# + zt approximate for faster results, maybe less accurate

        # Mamba Layer -2  ====================================================================
            
        u_2n = self.rms_2(u_2)
        But = self.B2(u_2n)
        delta_2 = F.softplus(self.dt_proj_a(u_2n)).expand(-1, self.hidden_dim)
        A_cont_2 = -torch.exp(self.A_log_2) # ( 1 x d)
        At_bar_2 = torch.exp(delta_2 * A_cont_2) # ( n x d)

        C_2 = (P_L_inv @ But * delta_2)[None]*self.t_weights[:,None,None]
        RHS_2 = torch.exp((delta_2 * A_cont_2)[None] * self.t_nodes[:,None,None] )
        integral_2 = torch.sum(LHS@C_2*RHS_2,dim=0)
        updated_c_2 = As_bar@(self.m2_vec[data.active_input_ids] * At_bar_2) + integral_2

        final_state = u_2 + F.gelu(updated_c_2,approximate='tanh') # approximate for faster results


        # Update states/ Embeddings

         # Update each state vector
        idx = data.active_input_ids  # indices [num_active]
        self.m1_vec = scatter_mean_update(self.m1_vec, idx, updated_c_1)
        
        if self.retain_message:
            self.old_message[data.active_input_ids] = x_in #.detach()

        if self.update_type == 'mamba': 
            self.m2_vec = scatter_mean_update(self.m2_vec, idx, updated_c_2)
        self.ne_vec = scatter_mean_update(self.ne_vec, idx, final_state)

    def BU_update(self,data):
        L_t = data.L_agg
        delta_L = data.delta_L_agg
         #+ reg**2 * L_t2 ( Linear Approximation )

        # Add Contextual Info to Signal

        neigh_info = torch.zeros((len(data.x_sub),2*self.embd_dims),device=self.device)
        neigh_info[:data.target_src.numel(),:self.embd_dims] = self.names(data.target_src)
        # neigh_info[:data.target_src.numel(),2*self.embd_dims:] = postive_time_enc
        neigh_info[:data.target_src.numel(),self.embd_dims:2*self.embd_dims] = self.names(data.target_dst)

        neigh_info[data.target_src.numel():2*data.target_src.numel(),:self.embd_dims] = self.names(data.target_dst)
        neigh_info[data.target_src.numel():2*data.target_src.numel(),self.embd_dims:2*self.embd_dims] = self.names(data.target_src)
        # neigh_info[data.target_src.numel():2*data.target_src.numel(),2*self.embd_dims:] = postive_time_enc
        # print(neigh_info.shape,data.x_sub.shape)
        x_in = torch.cat([data.x_sub,neigh_info],dim=1)

        if self.retain_message: # Use old message for neighbours nodes
            x_in[2*data.target_src.numel():] = self.old_message[data.active_input_ids[2*data.target_src.numel():]] #.detach()

        N, d = len(L_t), self.hidden_dim
        alpha = self.delta

        # Identity matrices (make sure on correct device)
        I_N = torch.eye(N, device=self.device)
        P_L_inv = I_N - self.reg*L_t
        I_d = torch.eye(d, device=self.device)
        C_k = self.m1_vec[data.active_input_ids]
        # Vectorize C_k and X_kp1
        c_k = C_k.reshape(-1, 1)          # (N*d, 1)
        zt = self.TuneInput(x_in)
        x_kp1 = zt.reshape(-1, 1)      # (N*d, 1)

        # Compute M matrix = - (A ⊗ I_N) - (I_d ⊗ L1*L2)
        # Compute L1_kp1 * L2_kp1
        L1L2 = torch.matmul(P_L_inv, self.reg*delta_L)  # (N, N)

        # Kronecker products
        kron_A = torch.kron(self.A_, I_N)       # (N*d, N*d)
        kron_L1L2 = torch.kron(I_d, L1L2) # (N*d, N*d)

        M_ = -kron_A - kron_L1L2           # (N*d, N*d)

        # Build system matrix: I - alpha * M
        I_big = torch.eye(N*d, device=self.device)
        system_matrix = I_big - alpha * M_ # (N*d, N*d)

        # Compute forcing term f = (B ⊗ L1) x_kp1
        kron_B_L1 = torch.kron(self.B_hippo, P_L_inv) # (N*d, N*d)
        f_kp1 = torch.matmul(kron_B_L1, x_kp1) # (N*d, 1)

        # Right-hand side
        rhs = c_k + alpha * f_kp1         # (N*d, 1)

        # Solve linear system for c_{k+1}
        c_kp1 = torch.linalg.solve(system_matrix, rhs)  # (N*d, 1)

        # Reshape back to (N, d)
        C_kp1 = c_kp1.reshape(N, d)

        # Update each state vector
        idx = data.active_input_ids  # indices [num_active]
        self.m1_vec = scatter_mean_update(self.m1_vec, idx, C_kp1)
        
        final_state = zt + F.gelu(C_kp1)
        if self.retain_message:
            self.old_message[data.active_input_ids] = x_in #.detach()
        self.ne_vec = scatter_mean_update(self.ne_vec, idx, final_state)

def zoh_update_HIPPO(self,data:active_data):
        L_t = data.L_agg
        Is = torch.eye(len(L_t), device=L_t.device)
        delta_L = data.delta_L_agg
        As_bar = Is - self.reg*delta_L + self.reg**2 * delta_L @ (L_t + delta_L/2)
        
        dt = delta_L[None] * self.t_nodes[:,None,None]
        #dt2 = dt@dt # (8,n,n)
        LHS = Is[None] - self.reg*dt + self.reg**2*(dt@(L_t[None] + dt/2)) #- reg**3 * dt @ (L_t2[None] + L_t[None]@dt + dt2/6) 
        P_L_inv = Is - self.reg*L_t #+ reg**2 * L_t2 ( Linear Approximation )

        # Add Contextual Info to Signal

        neigh_info = torch.zeros((len(data.x_sub),2*self.embd_dims),device=self.device)
        neigh_info[:data.target_src.numel(),:self.embd_dims] = self.names(data.target_src)
        # neigh_info[:data.target_src.numel(),2*self.embd_dims:] = postive_time_enc
        neigh_info[:data.target_src.numel(),self.embd_dims:2*self.embd_dims] = self.names(data.target_dst)

        neigh_info[data.target_src.numel():2*data.target_src.numel(),:self.embd_dims] = self.names(data.target_dst)
        neigh_info[data.target_src.numel():2*data.target_src.numel(),self.embd_dims:2*self.embd_dims] = self.names(data.target_src)
        # neigh_info[data.target_src.numel():2*data.target_src.numel(),2*self.embd_dims:] = postive_time_enc
        # print(neigh_info.shape,data.x_sub.shape)
        x_in = torch.cat([data.x_sub,neigh_info],dim=1)

        if self.retain_message: # Use old message for neighbours nodes
            x_in[2*data.target_src.numel():] = self.old_message[data.active_input_ids[2*data.target_src.numel():]] #.detach()

        zt = self.TuneInput(x_in) # input Tuning / Selective Scan
        ztn = self.rms_1(zt)
        Bzt = self.B1(ztn)
        # delta_1 = F.softplus(self.dt_proj_a(ztn)).expand(-1, self.hidden_dim) # (n x d) delta for each input
        # A_cont_1 = -torch.exp(self.A_log_1) # ( 1 x d)
        # At_bar_1 = torch.exp(delta_1 * A_cont_1) # ( n x d)

        At_bar_1 = torch.exp(self.delta*self.A_)

        C_1 = (P_L_inv @ (Bzt))[None]*self.t_weights[:,None,None] 
        RHS_1 = torch.exp((delta_1 * A_cont_1)[None] * self.t_nodes[:,None,None] )
        integral_1 = torch.sum(LHS@C_1*RHS_1,dim=0)

        updated_c_1= As_bar@(self.m1_vec[data.active_input_ids] * At_bar_1) + integral_1
        u_2 = zt+F.gelu(updated_c_1,approximate='tanh') #ztn #F.gelu(updated_c_1,approximate='tanh')# + zt approximate for faster results, maybe less accurate

        # Mamba Layer -2  ====================================================================
            
        u_2n = self.rms_2(u_2)
        But = self.B2(u_2n)
        delta_2 = F.softplus(self.dt_proj_a(u_2n)).expand(-1, self.hidden_dim)
        A_cont_2 = -torch.exp(self.A_log_2) # ( 1 x d)
        At_bar_2 = torch.exp(delta_2 * A_cont_2) # ( n x d)

        C_2 = (P_L_inv @ But * delta_2)[None]*self.t_weights[:,None,None]
        RHS_2 = torch.exp((delta_2 * A_cont_2)[None] * self.t_nodes[:,None,None] )
        integral_2 = torch.sum(LHS@C_2*RHS_2,dim=0)
        updated_c_2 = As_bar@(self.m2_vec[data.active_input_ids] * At_bar_2) + integral_2

        final_state = u_2 + F.gelu(updated_c_2,approximate='tanh') # approximate for faster results


        # Update states/ Embeddings

         # Update each state vector
        idx = data.active_input_ids  # indices [num_active]
        self.m1_vec = scatter_mean_update(self.m1_vec, idx, updated_c_1)
        
        if self.retain_message:
            self.old_message[data.active_input_ids] = x_in #.detach()

        if self.update_type == 'mamba': 
            self.m2_vec = scatter_mean_update(self.m2_vec, idx, updated_c_2)
        self.ne_vec = scatter_mean_update(self.ne_vec, idx, final_state)
