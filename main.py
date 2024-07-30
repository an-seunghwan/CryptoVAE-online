#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from datetime import datetime
import torch

import sys
sys.path.append('./modules')
import importlib
layers = importlib.import_module('modules.layers')
importlib.reload(layers)
utils = importlib.import_module('modules.utils')
importlib.reload(utils)
#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--model', type=str, default='GLD_finite', 
                        help='Fitting model options: GLD_finite')
    
    parser.add_argument("--d_latent", default=16, type=int,
                        help="size of latent dimension")
    parser.add_argument("--d_model", default=8, type=int,
                        help="size of transformer model dimension")
    parser.add_argument("--num_heads", default=1, type=int,
                        help="the number of heads in transformer")
    parser.add_argument("--num_layer", default=1, type=int,
                        help="the number of layers in transformer")
    parser.add_argument("--M", default=10, type=int,
                        help="the number of knot points")
    parser.add_argument("--K", default=20, type=int,
                        help="the number of quantiles to estimate")
    
    parser.add_argument("--timesteps", default=20, type=int, # equals to C
                        help="the number of conditional time steps")
    parser.add_argument("--future", default=1, type=int, # equals to T - C
                        help="the number of time steps to forecasting")
    parser.add_argument("--test_len", default=200, type=int,
                        help="length of test dataset")
    
    parser.add_argument("--MC", default=500, type=int,
                        help="the number of samples in Monte Carlo sampling")
    parser.add_argument('--epochs', default=400, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.0025, type=float,
                        help='learning rate')
    parser.add_argument('--threshold', default=1e-8, type=float,
                        help='threshold for clipping alpha_tilde')
    
    parser.add_argument('--prior_var', default=0.5, type=float,
                        help='variance of prior distribution')
    parser.add_argument('--beta', default=0.1, type=float,
                        help='scale parameter of asymmetric Laplace distribution')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    """load config"""
    config_path = f'./configs/{config["model"]}.yaml'
    if os.path.isfile(config_path):
        config = utils.load_config(config, config_path)
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if config["cuda"] else torch.device('cpu')

    utils.set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    dir_ = './assets/'
    if not os.path.exists(dir_): os.makedirs(dir_)
        
    """train, test split"""
    df = pd.read_csv(
        f'./data/data.csv',
        index_col=0
    )
    scaling_dict = {
        "BTC": 1e7,
        "ETH": 1e6,
        "XRP": 1e2,
        "ADA": 1e2,
        "ETC": 1e4,
        "XLM": 1e2,
        "BCH": 1e5,
    }
    
    colnames = [col.replace("KRW-", "") for col in df.columns.to_list()]
    config["p"] = df.shape[1]
    # reconstruct only T - C
    train_list, test_list = utils.build_datasets(df, config["test_len"], config)
    #%%
    """model"""
    try:
        model_module = importlib.import_module('modules.{}'.format(config["model"]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"])(config, device).to(device)
    except:
        model_module = importlib.import_module('modules.{}'.format(config["model"].split('_')[0]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"].split('_')[0])(config, device).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    model.train()
    #%%
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of Parameters:", num_params)
    #%%
    """Training"""
    try:
        train_module = importlib.import_module('modules.{}_train'.format(config["model"]))
    except:
        train_module = importlib.import_module('modules.{}_train'.format(config["model"].split('_')[0]))
    importlib.reload(train_module)
    
    train_context, train_target = train_list[1] 
    iterations = len(train_context) // config["batch_size"] + 1
        
    for e in range(config["epochs"]):
        logs = train_module.train_function(train_context, train_target, model, iterations, config, optimizer, device)
        
        if e % 10 == 0 or e == config["epochs"] - 1:
            print_input = "[EPOCH {:03d}/{}]".format(e + 1, config['epochs'])
            print_input += ''.join([', {}: {:.4f}'.format(x, y.item() / iterations) for x, y in logs.items()])
            print(print_input)
    #%%
    """model save"""
    torch.save(model.state_dict(), f'assets/{config["model"]}.pth')
    #%%
    """Quantile Estimation"""
    alphas = [0.1, 0.5, 0.9]
    train_context_ = torch.cat(
        [
            train_context, 
            torch.from_numpy(
                df.iloc[-config["timesteps"]:, :].values
            ).to(torch.float32)[None, ...]
        ], axis=0
    )
    est_quantiles = model.est_quantile(train_context_, alphas, config["MC"])
    estQ = [Q[:-1, ...][config["future"]::, :, :].reshape(-1, config["p"]) for Q in est_quantiles]
    target_ = train_target[config["future"]::, :, :].reshape(-1, config["p"])
    forecasts = [Q[-1] for Q in est_quantiles]
    #%%
    """Visualize"""
    fig = utils.visualize_quantile(target_, estQ, colnames, forecasts, scaling_dict, show=False)
    #%%
    """Forecasting results"""
    df_forecast = []
    for j, col in enumerate(colnames):
        df_forecast.append(
            [
                col, 
                forecasts[0][0, j].item(),
                forecasts[1][0, j].item(),
                forecasts[2][0, j].item(),
            ]
        )
    df_forecast = pd.DataFrame(df_forecast, columns=["names", "10%", "50%", "90%"])
    df_forecast.to_csv("assets/forecasting.csv") # tomorrow
    #%%
    try:
        history = pd.read_csv("assets/history.csv", index_col=0)
    except:
        history = pd.DataFrame()
    tomorrow = datetime.strftime(
        pd.to_datetime(df.index[-1]) + pd.Timedelta('24:00:00'), 
    '%Y-%m-%d %H:%M:%S')
    df_forecast["time"] = tomorrow
    history = pd.concat([history, df_forecast], axis=0)
    history.to_csv("assets/history.csv")
#%%
if __name__ == '__main__':
    main()
#%%