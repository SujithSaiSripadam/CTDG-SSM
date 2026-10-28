from ssm_utlis import *

class active_data:
    '''Hold acitve data for dataset'''
    def __init__(self) -> None:
        self.target_src = torch.empty(0)
        self.target_dst = torch.empty(0)
        self.target_enc = torch.empty(0)
        self.L_t = torch.empty(0)
        self.x_sub = torch.empty(0)
        self.neg_dst = torch.empty(0)
        self.t = None
        self.active_ids = torch.empty(0)
        self.active_input_ids = torch.empty(0)
        self.active_src_mask = torch.empty(0)
        self.active_dst_mask = torch.empty(0)
        self.active_neg_mask = torch.empty(0)
        self.L_agg = torch.empty(0)
        self.delta_L_agg = torch.empty(0)
        self.L_out = torch.empty(0)
        self.label_t = torch.empty(0)
        self.total = 0
        self.end_train = False
        pass
        
class TemporalEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, src, dst, times, edge_features,T,num_node,LastInteraction, context_window,inductive_cut = 1,biparted=False,device="cpu",task='LP',labels=None,unique_batch=False,negative='RNS',mode='Train',c_src=None,c_dst=None,c_time=None,inductive_mask = None,eval_seed = 42,test_edge_set=set()):
        self.src = torch.tensor(src, dtype=torch.long , device=device)
        self.dst = torch.tensor(dst, dtype=torch.long , device=device)
        self.times = torch.tensor(times, dtype=torch.float, device=device)
        

        if mode=='train':
            self.neighbour,self.ntimes,self.offset = build_neighbor_index_torch(self.src,self.dst,self.times,num_node,device=device)
        elif mode == 'eval' :
            self.random_seed = eval_seed
            self.inductive_mask = inductive_mask
            c_src = torch.tensor(c_src, dtype=torch.long, device=device)
            c_dst = torch.tensor(c_dst, dtype=torch.long, device=device)
            c_time = torch.tensor(c_time, dtype=torch.float, device=device)
            
            self.neighbour,self.ntimes,self.offset = build_neighbor_index_torch(c_src,c_dst,c_time,num_node,device=device)
        


        # self.uni_timestamps = torch.unique(self.times)
        self.edge_features = torch.tensor(edge_features, dtype=torch.float,device=device)
        self.device = device
        self.LastInteraction=torch.tensor(LastInteraction, dtype=torch.float, device=device)
        # self.timestamps = torch.unique(self.times).sort()[0]
        self.T = T
        self.biparted = biparted
        self.num_node = num_node
        self.context_window = context_window
        self.n_mode = mode
        self.task = task
        self.unique_batch = unique_batch
        self.negative = negative
        # speedup #2 (INFERENCE_SPEEDUP_PLAN.md §5): per-window CPU allocations
        # cached once. t_nodes cache = arange(2T) (windows are ≤ T edges, so
        # 2*len(src_t) ≤ 2T; lazy re-extend if ever exceeded). Values identical
        # per window (slice of arange == arange).
        self._t_nodes_cache = None
        self._inductive_mask_gpu = None
        if self.biparted:
            self.rnd_dst = torch.zeros(self.num_node, dtype=torch.bool, device=device)
            self.rnd_dst[self.dst]=True
            self.rnd_dst = self.rnd_dst.nonzero(as_tuple=True)[0]

        if self.n_mode == 'eval':
            self.test_edge_set = test_edge_set

        if self.task == 'LP':
            self.historical_edge = set()

        if unique_batch :
            self.start,self.end = batch_indices_fast(self.src,self.dst,B=self.T,num_nodes=num_node)

        if task == 'NC':
            self.labels = torch.tensor(labels, dtype=torch.float,device=device) #[:len(times)]



    def __len__(self):
        if self.unique_batch:
            return len(self.start)
        else :
            return len(self.times)

    def __getitem__(self, idx)->active_data:
        data = active_data()
        data.end_train = False
        if self.unique_batch:
            t_start = self.start[idx]
            t_end = self.end[idx]
        
        else :
            t_start = idx
            t_end = idx+self.T
            
            if t_end>=len(self.times):
                data.end_train=True
                t_end = len(self.times)-1
            elif t_end==len(self.times)-1:
                data.end_train=True



        tau = self.times[t_start]
        src_t = self.src[t_start:t_end]
        dst_t = self.dst[t_start:t_end]
        
        
        if self.n_mode=='eval' and self.task == 'LP':
            g = torch.Generator(device=self.device)
            g.manual_seed(idx+len(self.times))
            if self._inductive_mask_gpu is None:   # one-time H2D (was: torch.tensor(...) per window)
                self._inductive_mask_gpu = torch.as_tensor(self.inductive_mask, device=self.device)
            data.inductive_mask = self._inductive_mask_gpu[t_start:t_end]
        

        total = 2*len(src_t)
        if self._t_nodes_cache is None or self._t_nodes_cache.numel() < total:
            self._t_nodes_cache = torch.arange(max(total, 2*self.T), device=self.device)
        t_nodes_id = self._t_nodes_cache[:total]
        combined_nodes = torch.hstack([src_t,dst_t])

        if self.task == 'NC':
            data.label_t = self.labels[t_start:t_end].to(self.device)
            data.src_t = src_t
            data.dst_t = dst_t

        time_t = self.times[t_start:t_end].to(self.device)
        last_t = self.LastInteraction[t_start:t_end]
        # enc_t = self.enc_time[self.t_start:t_end]
        edge_feats_t = self.edge_features[t_start:t_end]


        data.target_src = src_t
        data.target_dst = dst_t
        data.target_enc = last_t #time_t #enc_t
        collide_check = True
        # Negative sampling (vectorized)
        if self.task == 'LP':
            E = src_t.shape[0]
            # Draw k candidates for each edge
            if self.n_mode == 'eval':
                u,v = src_t.tolist(),dst_t.tolist()
                batch_edge = set((ub,vb) for ub,vb in zip(u,v))
                hist_valid_edge = torch.tensor(list(self.historical_edge - batch_edge),dtype=torch.long, device=self.device).t()
                ind_valid_edge = torch.tensor(list((self.historical_edge - self.test_edge_set) - batch_edge),dtype=torch.long, device=self.device).t()
                
                size_EH = hist_valid_edge.shape[-1]
                if E >size_EH:
                    candidates = torch.randint(0, self.num_node, (E-size_EH, 5), device=src_t.device,generator=g)
                    invalid = (candidates == src_t[size_EH:, None]) | (candidates == dst_t[size_EH:, None])
                    candidates = candidates.masked_fill(invalid, -1)
                    # For each row, find the *first* valid candidate
                    valid_mask = candidates != -1
                    idx = valid_mask.float().argmax(dim=1)   # position of first True
                    rnd_neg_dst = candidates[torch.arange(E-size_EH, device=src_t.device), idx]
                    if size_EH>0:
                        data.hist_neg_src,data.hist_neg_dst = hist_valid_edge
                        # add random edge
                        data.hist_neg_src = torch.hstack([data.hist_neg_src,src_t[size_EH:]])
                        data.hist_neg_dst = torch.hstack([data.hist_neg_dst,rnd_neg_dst])
                    else :
                        data.hist_neg_dst = rnd_neg_dst
                        data.hist_neg_src = src_t


                else:
                    # print('Used Historic Edge')
                    perm = torch.randperm(size_EH, device=self.device,generator=g)[:E]
                    data.hist_neg_src,data.hist_neg_dst = hist_valid_edge[:,perm]

                data.hist_neg_t = get_delta_t(data.hist_neg_src,data.hist_neg_dst,time_t,self.neighbour,self.ntimes,self.offset,default=1e+11)

                size_EI = ind_valid_edge.shape[-1]
                if E >size_EI:
                    candidates = torch.randint(0, self.num_node, (E-size_EI, 5), device=src_t.device,generator=g)
                    invalid = (candidates == src_t[size_EI:, None]) | (candidates == dst_t[size_EI:, None])
                    candidates = candidates.masked_fill(invalid, -1)
                    # For each row, find the *first* valid candidate
                    valid_mask = candidates != -1
                    idx = valid_mask.float().argmax(dim=1)   # position of first True
                    rnd_neg_dst = candidates[torch.arange(E-size_EI, device=src_t.device), idx]
                    if size_EI>0:
                        data.ind_neg_src,data.ind_neg_dst = ind_valid_edge
                        # add random edge
                        data.ind_neg_src = torch.hstack([data.ind_neg_src,src_t[size_EI:]])
                        data.ind_neg_dst = torch.hstack([data.ind_neg_dst,rnd_neg_dst])
                    else:
                        data.ind_neg_src=src_t
                        data.ind_neg_dst = rnd_neg_dst

                else:
                    # print('Used Inductive Edge')
                    perm = torch.randperm(size_EI, device=self.device,generator=g)[:E]
                    data.ind_neg_src,data.ind_neg_dst = ind_valid_edge[:,perm]

                data.ind_neg_t = get_delta_t(data.ind_neg_src,data.ind_neg_dst,time_t,self.neighbour,self.ntimes,self.offset,default=1e+11)
                candidates = torch.randint(0, self.num_node, (E, 5), device=src_t.device)


            elif self.n_mode == 'train':
                if self.biparted:
                    rand_idx = torch.randint(0, self.rnd_dst.shape[0], (E, 5), device=src_t.device)
                    candidates = self.rnd_dst[rand_idx] #torch.randint(0, self.num_node, (E, 5), device=src_t.device)
                else:
                    candidates = torch.randint(0, self.num_node, (E, 5), device=src_t.device)

                
            if collide_check :    
                # Mark invalid ones (elementwise check)
                invalid = (candidates == src_t[:, None]) | (candidates == dst_t[:, None])
                candidates = candidates.masked_fill(invalid, -1)
                
                # For each row, find the *first* valid candidate
                valid_mask = candidates != -1
                idx = valid_mask.float().argmax(dim=1)   # position of first True
                neg_dst = candidates[torch.arange(E, device=src_t.device), idx]

                # Rare case: all k candidates invalid → resample safely
                mask = neg_dst == -1
                if mask.any():
                    print('Chnage Seed')
                    resample = torch.randint(0, self.num_node, (mask.sum().item(),), device=src_t.device)
                    bad = (resample == src_t[mask]) | (resample == dst_t[mask])
                    while bad.any():
                        resample[bad] = torch.randint(0, self.num_node, (bad.sum().item(),), device=src_t.device)
                        bad = (resample == src_t[mask]) | (resample == dst_t[mask])
                    neg_dst[mask] = resample
                data.neg_dst = neg_dst
                data.neg_src = src_t

            data.rnd_neg_t = get_delta_t(data.neg_src,data.neg_dst,time_t,self.neighbour,self.ntimes,self.offset,default=1e+11)
            # print(data.target_enc_neg,data.target_enc)


        # if self.negative == 'HNS': # Add Historical Edges
        if self.task == 'LP':
            u,v = src_t.tolist(),dst_t.tolist()
            self.historical_edge.update((ue,ve) for ue,ve in zip(u,v))

        com_neighbours = temporal_sample_neighbors(combined_nodes,self.neighbour,self.ntimes,self.offset,tau=tau,k=self.T)
        com_mask = com_neighbours>-1
        com_neighbours = com_neighbours[com_mask]
        com_src = t_nodes_id[:,None].repeat(1,self.T)[com_mask]
        # com_src = t_nodes_id
        un_neigh,com_neigh = torch.unique(com_neighbours,return_inverse=True)
        com_neigh += total # shift for active nodes 
        data.x_sub = torch.zeros((total+len(un_neigh),edge_feats_t.shape[-1]),device=self.device)
        data.x_sub[:len(src_t),:] = edge_feats_t
        data.x_sub[len(src_t):2*len(src_t)] = edge_feats_t
        data.active_input_ids = torch.hstack([combined_nodes,un_neigh])

        if self.task == 'LP':
            data.active_ids = torch.hstack([combined_nodes,data.neg_src,data.neg_dst])

        com_edge_index_src = torch.hstack([t_nodes_id,com_src])
        com_edge_index_dst = torch.hstack([t_nodes_id.flip(0),com_neigh])

        data.L_agg =  build_laplacian_windowed_agg(com_edge_index_src, com_edge_index_dst, total+len(un_neigh) ,None, self.device)
        data.total = total
        delta_L = torch.zeros_like(data.L_agg,device=data.L_agg.device)
        delta_L[:total,:total] = data.L_agg[:total,:total]
        # delta_L = build_laplacian_windowed_agg(input_inverse[:src_t.numel()],input_inverse[all_src.numel():all_src.numel()+dst_t.numel()], len(data.active_input_ids),None, self.device)
        data.delta_L_agg = delta_L#-torch.eye(len(delta_L)).to(self.device)

        return data