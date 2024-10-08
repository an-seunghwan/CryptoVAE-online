#%%
import numpy as np
import tqdm
import torch
# from joblib import Parallel, delayed
#%%
def train_function(context, target, model, iterations, config, optimizer, device):
    tau = torch.linspace(0.01, 0.99, config["K"]).unsqueeze(0).to(device)
    
    logs = {
        'loss': 0,
        'quantile': 0,
        'KL': 0,
        'active': 0,
    }
    for i in range(iterations):
        idx = np.random.choice(
            range(len(context)), config["batch_size"], replace=False)
        context_batch = context[idx, :].to(device)
        target_batch = target[idx, :].to(device)
        
        optimizer.zero_grad()
        
        prior, posterior, params = model(context_batch, target_batch)
        
        """Quantile""" # broadcasting version
        target_batch_ = target_batch.reshape(-1, config["p"]).permute(1, 0).reshape(-1, 1)
        
        theta1 = torch.cat([torch.cat(params[i][0], dim=1)[:, None, :] for i in range(len(params))], dim=1) # i = time
        theta2 = torch.cat([torch.cat(params[i][1], dim=1)[:, None, :] for i in range(len(params))], dim=1) # i = time
        theta3 = torch.cat([torch.cat(params[i][2], dim=1)[:, None, :] for i in range(len(params))], dim=1) # i = time
        theta4 = torch.cat([torch.cat(params[i][3], dim=1)[:, None, :] for i in range(len(params))], dim=1) # i = time
        
        theta1 = theta1.reshape(-1, config["p"]).permute(1, 0).reshape(-1, 1)
        theta2 = theta2.reshape(-1, config["p"]).permute(1, 0).reshape(-1, 1)
        theta3 = theta3.reshape(-1, config["p"]).permute(1, 0).reshape(-1, 1)
        theta4 = theta4.reshape(-1, config["p"]).permute(1, 0).reshape(-1, 1)
        
        Q = model.quantile_function(tau, theta1, theta2, theta3, theta4)
        residual = target_batch_ - Q
        quantile = (residual * (tau - (residual < 0).to(torch.float32))).sum()
        quantile /= (config["K"] * context_batch.size(0))
        logs["quantile"] = logs.get("quantile") + quantile
        
        """KL-divergence"""
        prior_mean = torch.cat([torch.cat(prior.mean[i], dim=0) for i in range(len(prior.mean))], dim=0)
        prior_logvar = torch.cat([torch.cat(prior.logvar[i], dim=0) for i in range(len(prior.logvar))], dim=0)
        posterior_mean = torch.cat([torch.cat(posterior.mean[i], dim=0) for i in range(len(posterior.mean))], dim=0)
        posterior_logvar = torch.cat([torch.cat(posterior.logvar[i], dim=0) for i in range(len(posterior.logvar))], dim=0)
        
        KL = ((posterior_mean - prior_mean).pow(2) / prior_logvar.exp()).sum(dim=1)
        KL += (prior_logvar - posterior_logvar).sum(dim=1)
        KL += (posterior_logvar.exp() / prior_logvar.exp()).sum(dim=1)
        KL -= config["d_latent"]
        KL *= 0.5
        KL = KL.sum() / context_batch.size(0)
        logs["KL"] = logs.get("KL") + KL
        
        loss = quantile + config["beta"] * KL
        logs["loss"] = logs.get("loss") + loss
        
        active = (posterior_logvar.exp().mean(dim=0) < 0.1).to(torch.float32).mean()
        logs["active"] = logs.get("active") + active
        
        loss.backward()
        optimizer.step()
        
    return logs
#%%
# quantile = (residual * (tau - (residual < 0).to(torch.float32))) 
# quantile /= torch.repeat_interleave(maxvalues.t(), context_batch.size(0) * config["future"], dim=0)
# quantile = quantile.sum() / (config["K"] * context_batch.size(0))
#%%
# gamma = torch.cat([torch.cat(params[i][0], dim=0) for i in range(len(params))], dim=0)
# beta = torch.cat([torch.cat(params[i][1], dim=0) for i in range(len(params))], dim=0)
# delta = torch.cat([torch.cat(params[i][2], dim=0) for i in range(len(params))], dim=0)
#%%
# j = 0 # coin
# quantile_sum = 0
# for j in range(target_batch.size(-1)):
#     target_batch_ = target_batch[..., j].reshape(-1, 1)
    
#     theta1 = torch.cat([params[i][0][j][:, None, :] for i in range(len(params))], dim=1) # i = time
#     theta2 = torch.cat([params[i][1][j][:, None, :] for i in range(len(params))], dim=1) # i = time
#     theta3 = torch.cat([params[i][2][j][:, None, :] for i in range(len(params))], dim=1) # i = time
#     theta4 = torch.cat([params[i][3][j][:, None, :] for i in range(len(params))], dim=1) # i = time
    
#     theta1 = theta1.reshape(-1, theta1.size(-1))
#     theta2 = theta2.reshape(-1, theta2.size(-1))
#     theta3 = theta3.reshape(-1, theta3.size(-1))
#     theta4 = theta4.reshape(-1, theta4.size(-1))
    
#     Q = model.quantile_function(tau, theta1, theta2, theta3, theta4)
#     residual = target_batch_ - Q
#     quantile = (residual * (tau - (residual < 0).to(torch.float32))).sum()
#     quantile /= (config["K"] * context_batch.size(0))
#     quantile_sum += quantile
# logs["quantile"] = logs.get("quantile") + quantile_sum
#%%
# def optim(j, model, target_batch, context_batch, params, tau, config):
#     target_batch_ = target_batch[..., j].reshape(-1, 1)
    
#     theta1 = torch.cat([params[i][0][j][:, None, :] for i in range(len(params))], dim=1) # i = time
#     theta2 = torch.cat([params[i][1][j][:, None, :] for i in range(len(params))], dim=1) # i = time
#     theta3 = torch.cat([params[i][2][j][:, None, :] for i in range(len(params))], dim=1) # i = time
#     theta4 = torch.cat([params[i][3][j][:, None, :] for i in range(len(params))], dim=1) # i = time
    
#     theta1 = theta1.reshape(-1, theta1.size(-1))
#     theta2 = theta2.reshape(-1, theta2.size(-1))
#     theta3 = theta3.reshape(-1, theta3.size(-1))
#     theta4 = theta4.reshape(-1, theta4.size(-1))
    
#     Q = model.quantile_function(tau, theta1, theta2, theta3, theta4)
#     residual = target_batch_ - Q
#     quantile = (residual * (tau - (residual < 0).to(torch.float32))).sum()
#     quantile /= (config["K"] * context_batch.size(0))
#     return quantile

# results = Parallel(n_jobs=config["p"])(
#     delayed(optim)(j, model, target_batch, context_batch, params, tau, config) 
#     for j in range(config["p"]))
# quantile = sum(results)
#%%