import torch
import torch.nn.functional as func


# vanilla_logits := logits w/o att_ block
# logits := logits w/ att_ block
def disimilarity_entropy(logits, vanilla_logits, t=1.):
    n_prob = torch.clamp(torch.softmax(vanilla_logits, dim=1), min=1e-7)
    a_prob = torch.clamp(torch.softmax(logits, dim=1), min=1e-7)

    n_entropy = -torch.sum(n_prob * torch.log(n_prob), dim=1) / t
    a_entropy = -torch.sum(a_prob * torch.log(a_prob), dim=1) / t

    entropy_disimilarity = torch.nn.functional.mse_loss(input=a_entropy, target=n_entropy, reduction="none")
    assert ~torch.isnan(entropy_disimilarity).any(), print(torch.min(n_entropy), torch.max(a_entropy))

    return entropy_disimilarity


def energy_loss(logits, targets, vanilla_logits, out_idx=254, t=1.): #batch*19*700*700
    out_msk = (targets == out_idx)
    void_msk = (targets == 255)
    back_mask = (targets == 100)

    pseudo_targets = torch.argmax(vanilla_logits, dim=1)
    outlier_msk = (out_msk | void_msk| back_mask)
    a_prob = torch.clamp(torch.softmax(logits, dim=1), min=1e-7)
    log_a_prob=torch.log(a_prob)
    t_prob= torch.clamp(torch.softmax(vanilla_logits, dim=1), min=1e-7)
    kl_loss = func.kl_div(log_a_prob, t_prob, reduction='none')
    kl_loss=torch.sum(kl_loss,dim=1)[~outlier_msk]
    entropy_part = func.cross_entropy(input=logits, target=pseudo_targets, reduction='none')[~outlier_msk]
    entropy_part=kl_loss*0.5+entropy_part*0.5
    reg = disimilarity_entropy(logits=logits, vanilla_logits=vanilla_logits)[~outlier_msk]
    

    a_out_entropy = -(1/19) * torch.log(a_prob+0.0001)
    a_out_entropy=torch.sum(a_out_entropy,dim=1)
    a_out_entropy=a_out_entropy[outlier_msk]
    lambd=0
    
    
    if torch.sum(out_msk) > 0:
        a_prob = torch.clamp(torch.softmax(logits, dim=1), min=1e-7)
        # a_prob_max=a_prob.max(dim=1)[0].flatten(start_dim=1)
        a_entropy = -(a_prob) * torch.log(a_prob+0.001)
        entorpy_score = torch.sum(a_entropy, dim=1)
        energy_outpart=torch.tensor([.0], device=targets.device)
        entropy_outpart=torch.tensor([.0], device=targets.device)
        logits = logits.flatten(start_dim=2).permute(0, 2, 1)
        if torch.sum(back_mask) > 0:
            energy_outpart= 0.5*torch.nn.functional.relu(-torch.log(torch.sum(torch.exp(logits),
                                                                   dim=2))[back_mask.flatten(start_dim=1)]).mean() 
            entropy_outpart=0.5*(entorpy_score.flatten(start_dim=1)[back_mask.flatten(start_dim=1)]).mean()
        entorpy_score=-(entorpy_score.flatten(start_dim=1)[out_msk.flatten(start_dim=1)]).mean()+(entorpy_score.flatten(start_dim=1)
                                        [~outlier_msk.flatten(start_dim=1)]).mean()+entropy_outpart*1

        
        
        # energy_part_1 = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits),
        #                                                            dim=2))[out_msk.flatten(start_dim=1)]).mean()

                                                        
                            
        energy_part_1 = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits),
                                                                   dim=2))[out_msk.flatten(start_dim=1)]).mean()+1*torch.nn.functional.relu(-torch.log(torch.sum(torch.exp(logits),
                                                                   dim=2))[~outlier_msk.flatten(start_dim=1)]).mean() + energy_outpart*1 #这里原本是outmask不包含void，
        
        ################### this is the weighted loss
        # energy_part_1 = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits),
        #                                                            dim=2)*a_prob_max)[out_msk.flatten(start_dim=1)]).mean()+torch.nn.functional.relu(-torch.log(torch.sum(torch.exp(logits),
        #                                                            dim=2)*a_prob_max)[~out_msk.flatten(start_dim=1)]).mean()
                                                                   
        energy_part=(energy_part_1+1*entorpy_score)/2      
        # energy_part= energy_part_1                                      
    else:
        energy_part = torch.tensor([.0], device=targets.device)
    return {"entropy_part": (1-lambd)*entropy_part.mean()+lambd*a_out_entropy.mean(), "reg": reg.mean(), "energy_part": energy_part}


