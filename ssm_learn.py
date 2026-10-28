from ssm_memory import *
from ssm_dataset import TemporalEdgeDataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from ssm_utlis import EarlyStopping,print_eval_metrics
from torch.utils.data import DataLoader

def collate_active_data(batch):
    return batch[0]  # each batch has size 1

def link_prediction_train_single_sequence(model:MemoryModel,dataset:TemporalEdgeDataset, optimizer:torch.optim.Adam, scaler=None):
    # ===============================================================================================================================================

    #model = MemoryModel(num_nodes=total_nodes, input_dim=edge_feat_dim + time_enc_dim, hidden_dim=64, delta=0.1, device='cuda').to('cuda')
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    total_loss = 0
        
    all_preds, all_labels = [], []

    # Only trained at window nodes! 
    t_warm = 100

    batch_loss = 0 
    # Reset Internal States 
    model.reset()
    batch_counter = 0
    batch_size = 10
    valid_grad = False
    unique_batch = dataset.unique_batch
    if unique_batch:
        counter = range(len(dataset))
    else:
        counter = range(0,len(dataset),dataset.T)

    # Use batch_size=1 if each item is already a “window”
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,collate_fn=collate_active_data)


    for t in tqdm(counter,desc='Epoch Progress: ',disable=False):
        data = dataset[t]


        # Predict State ==========================================================================================================
 
        states = model.ne_vec[data.active_ids]
        postive_time_enc = model.time_encoder(data.target_enc)
        negative_time_enc = model.time_encoder(data.rnd_neg_t)
  
        # memory[model.hidden_dim:] = (model.alpha*torch.eye(len(L_out)) + model.beta*L_out)@memory[model.hidden_dim:]
        if t>t_warm  :
            states_target_src = torch.hstack([model.names(data.target_src),states[:data.target_src.numel()]])
            states_target_dst = torch.hstack([model.names(data.target_dst),states[data.target_src.numel():2*data.target_src.numel()]])
            states_neg_src = torch.hstack([model.names(data.neg_src),states[2*data.target_src.numel():3*data.target_src.numel()]])
            states_neg_dst = torch.hstack([model.names(data.neg_dst),states[3*data.target_src.numel():]])
            # print(data.neg_dst)
            concat_states = torch.hstack([states_target_src,states_target_dst,postive_time_enc])
            pos_score = model.PredictionMap(concat_states) 
            pos_label = torch.ones_like(pos_score)
            concat_states_neg = torch.hstack([states_neg_src,states_neg_dst,negative_time_enc])
            neg_score = model.PredictionMap(concat_states_neg)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            label = torch.cat([pos_label, neg_label])
            loss = loss_fn(score, label) 
            valid_grad = True
            batch_loss += loss
            all_preds.append(score.detach())      # GPU; ONE .cpu() at end
            all_labels.append(label.detach())

        # ==========================================================================================================================
        # # Old Model ===================================================================================================================
        # if model.update_type =='HiPPO':
        #     L_t = data.L_agg
        #     reg = model.reg
        #     Is = torch.eye(len(L_t)).to(L_t.device)
        #     delta_L = data.delta_L_agg

        #     Id = torch.eye(len(model.A_hippo)).to(model.device)

        #     neigh_info = torch.zeros((len(data.x_sub),2*model.embd_dims),device=model.device)
        #     neigh_info[:data.target_src.numel(),:model.embd_dims] = model.names(data.target_src)
        #     neigh_info[data.target_src.numel():2*data.target_src.numel(),:model.embd_dims] = model.names(data.target_src)
        #     neigh_info[:data.target_src.numel(),model.embd_dims:] = model.names(data.target_dst)
        #     neigh_info[data.target_src.numel():2*data.target_src.numel(),model.embd_dims:] = model.names(data.target_dst)

        #     # print(neigh_info.shape,data.x_sub.shape)
        #     x_in = torch.cat([data.x_sub,neigh_info],dim=1)
        #     x_in[2*data.target_src.numel():] = model.old_message[data.active_input_ids[2*data.target_src.numel():]]
        #     zt = model.TuneInput_src(x_in) # input Tuning / Selective Scan
        #     delta_1 = F.softplus(model.dt_proj_a(zt)).expand(-1, model.hidden_dim)
            
        #     updated_c_1 =  (( prev_c_1[:,None,:] @ (Id[None,:,:]- delta_1[:,:,None] * model.A_hippo[None,:,:])).squeeze(1)
        #         +  (Is-(reg*L_t)) @ zt * delta_1
        #         - (Is-(reg*L_t) ) @ (reg*delta_L) @ prev_c_1)
            
        #     updated_c_1 = torch.clamp(updated_c_1,min=-1e+11,max=1e+11)

        #     final_state = updated_c_1 # (zt + F.gelu(updated_c_1,approximate='tanh'))


        # # Update Sates (Dual Mamba - 1 ) ============================================================================================
        
        model.zoh_update(data)
        # model.BU_update(data)
        #=========================================================================================================================
        if t>t_warm:
            batch_counter +=1

        if  (valid_grad and t>t_warm and batch_counter>= batch_size) or (valid_grad and data.end_train) :
            batch_counter=0
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            model.detach()
            valid_grad = False
            total_loss += batch_loss.item()
            batch_loss = 0

  
    y_pred = torch.cat(all_preds).cpu().numpy()
    y_true = torch.cat(all_labels).cpu().numpy()
    auc = roc_auc_score(y_true, y_pred)
    return total_loss, auc


def link_prediction_eval(model: MemoryModel, dataset: TemporalEdgeDataset,tq_mode=True):
    # Containers for all predictions and labels per case
    t_preds = {'rnd': [], 'hst': [], 'ind': []}
    t_labels = {'rnd': [], 'hst': [], 'ind': []}
    i_preds = {'rnd': [], 'hst': [], 'ind': []}
    i_labels = {'rnd': [], 'hst': [], 'ind': []}

    unique_batch = dataset.unique_batch
    if unique_batch:
        counter = range(len(dataset))
    else:
        counter = range(0, len(dataset) - 2 * dataset.T, dataset.T)

    for t in tqdm(counter, desc='Eval Progress', disable=tq_mode):
        data = dataset[t]

        # Encode times
        positive_time_enc = model.time_encoder(data.target_enc)
        negative_time_enc_rnd = model.time_encoder(data.rnd_neg_t)
        negative_time_enc_hst = model.time_encoder(data.hist_neg_t)
        negative_time_enc_ind = model.time_encoder(data.ind_neg_t)

        # Prepare positive states
        states_target_src = torch.hstack([model.names(data.target_src), model.ne_vec[data.target_src]])
        states_target_dst = torch.hstack([model.names(data.target_dst), model.ne_vec[data.target_dst]])

        # Prepare negative states (same neg_src and neg_dst for all negative types)
        states_neg_src = torch.hstack([model.names(data.neg_src), model.ne_vec[data.neg_src]])
        states_neg_dst = torch.hstack([model.names(data.neg_dst), model.ne_vec[data.neg_dst]])

        # Positive scores
        concat_states_pos = torch.hstack([states_target_src, states_target_dst, positive_time_enc])
        pos_score = model.PredictionMap(concat_states_pos)
        pos_label = torch.ones_like(pos_score)

        # Negative scores for each neg type
        concat_states_rnd = torch.hstack([states_neg_src, states_neg_dst, negative_time_enc_rnd])
        rnd_score = model.PredictionMap(concat_states_rnd)
        rnd_label = torch.zeros_like(rnd_score)

        concat_states_hst = torch.hstack([states_neg_src, states_neg_dst, negative_time_enc_hst])
        hst_score = model.PredictionMap(concat_states_hst)
        hst_label = torch.zeros_like(hst_score)

        concat_states_ind = torch.hstack([states_neg_src, states_neg_dst, negative_time_enc_ind])
        ind_score = model.PredictionMap(concat_states_ind)
        ind_label = torch.zeros_like(ind_score)

        # Inductive mask (pre-sliced GPU buffer in dataset — was torch.tensor(...).to(device) per window)
        i_mask = data.inductive_mask

        # Transductive masks (~i_mask)
        for neg_type, neg_score, neg_label in zip(
            ['rnd', 'hst', 'ind'],
            [rnd_score, hst_score, ind_score],
            [rnd_label, hst_label, ind_label]
        ):
            t_preds[neg_type].append(torch.cat([pos_score[~i_mask], neg_score[~i_mask]]).detach())
            t_labels[neg_type].append(torch.cat([pos_label[~i_mask], neg_label[~i_mask]]).detach())

        # Inductive masks (i_mask)
        for neg_type, neg_score, neg_label in zip(
            ['rnd', 'hst', 'ind'],
            [rnd_score, hst_score, ind_score],
            [rnd_label, hst_label, ind_label]
        ):
            i_preds[neg_type].append(torch.cat([pos_score[i_mask], neg_score[i_mask]]).detach())
            i_labels[neg_type].append(torch.cat([pos_label[i_mask], neg_label[i_mask]]).detach())

        model.zoh_update(data)

    # Compute metrics per case
    results = {'transductive': {}, 'inductive': {}}
    for neg_type in ['rnd', 'hst', 'ind']:
        # Concatenate all batches (single GPU→CPU transfer per pass)
        t_pred = torch.cat(t_preds[neg_type]).cpu().numpy()
        t_true = torch.cat(t_labels[neg_type]).cpu().numpy()
        i_pred = torch.cat(i_preds[neg_type]).cpu().numpy()
        i_true = torch.cat(i_labels[neg_type]).cpu().numpy()

        # Compute AUC and AP
        results['transductive'][neg_type] = {
            'AUC': roc_auc_score(t_true, t_pred),
            'AP': average_precision_score(t_true, t_pred)
        }
        results['inductive'][neg_type] = {
            'AUC': roc_auc_score(i_true, i_pred),
            'AP': average_precision_score(i_true, i_pred)
        }

    return results
    
    
def main(model, test_dataset,val_dataset,train_dataset, optimizer, num_epochs=10,name='trn', device='cuda'):
    path = name +  'best_model_nb.pt'
    early_stopping = EarlyStopping(patience=5, verbose=True, path=path,delta=0)
    scaler = None # amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.75, patience=3
    )

    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        loss, auc = link_prediction_train_single_sequence(model,train_dataset,optimizer,scaler)
        print(f'Train Loss: {loss}| Train auc: {auc}')
        with torch.no_grad():
            val_results = link_prediction_eval(model, val_dataset,tq_mode=False)

        print_eval_metrics(val_results, prefix="VAL")

        val_auc_t = sum(val_results['transductive'][k]['AUC'] for k in val_results['transductive']) / 3
        val_auc_i = sum(val_results['inductive'][k]['AUC'] for k in val_results['inductive']) / 3

        # print(torch.cuda.memory_summary())
        scheduler.step(val_auc_t + val_auc_i)
        print("Last LR:", scheduler.get_last_lr())
        early_stopping(val_auc_t + val_auc_i, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    torch.cuda.empty_cache()
    with torch.no_grad():
        # model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        results = link_prediction_eval(model,test_dataset,tq_mode=False)

    print_eval_metrics(results, prefix="TEST")
    return results

        

   
def ablate(model, test_dataset,val_dataset,train_dataset, optimizer, num_epochs=10,name='trn', device='cuda'):
    path = name +  'best_model_nb.pt'
    early_stopping = EarlyStopping(patience=10, verbose=True, path=path,delta=0.01)
    scaler = None # amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.75, patience=3
    )

    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        loss, auc = link_prediction_train_single_sequence(model,train_dataset,optimizer,scaler)
        print(f'Train Loss: {loss}| Train auc: {auc}')
        with torch.no_grad():
            val_results = link_prediction_eval(model, val_dataset,tq_mode=True)

        # print_eval_metrics(val_results, prefix="VAL")

        val_auc_t = sum(val_results['transductive'][k]['AUC'] for k in val_results['transductive']) / 3
        val_auc_i = sum(val_results['inductive'][k]['AUC'] for k in val_results['inductive']) / 3

        # print(torch.cuda.memory_summary())
        scheduler.step(val_auc_t + val_auc_i)
        # print("Last LR:", scheduler.get_last_lr())
        early_stopping(val_auc_t + val_auc_i, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    torch.cuda.empty_cache()
    with torch.no_grad():
        # model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        results = link_prediction_eval(model,test_dataset,tq_mode=False)

    # print_eval_metrics(results, prefix="TEST")
    return results

    