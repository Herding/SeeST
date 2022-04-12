import yaml

from torch.optim import Adam
from torch.nn import L1Loss
from models.graphwave import GraphWave
from utils.inits import seed_init, device_init
from utils.predictor import Predictor
from utils.supervisor import Generator, Incidentor
from utils.evaluator import mae, rmse, mape
from utils.loger import Loger


def run():
    with open('./config/GraphWave_METR.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file.read(), yaml.SafeLoader)

    seed_init(config['seed'])

    loger = Loger(config['model_name'], config['log_path'])
    device = device_init()

    #data
    incidentor = Incidentor(config['adj_path'], loger, device)
    adj = incidentor.adj

    if config['is_random_adj']:
        adj_init = None
    else:
        adj_init = adj

    generator = Generator(config['data_path'], loger, config['windows'], config['lag'],
                          config['horizon'], config['train_ratio'], config['val_ratio'],
                          config['bs'], config['is_scaler'], device)
    
    # model and for training
    opt = Adam
    loss = L1Loss()
    model = GraphWave(device, generator.train_data.nodes, config['droprate'], [adj, adj],
                      config['has_gcn'], config['has_apt_adj'], adj_init, config['d_in'],
                      config['d_out'], config['residual_channels'], config['dilation_channels'],
                      config['cat_feat_gc'], config['skip_channels'], config['end_channels'],
                      config['kernel_size'], config['units'], config['layers'], config['apt_size'])
    metrics = [mae, rmse, mape]

    predictor = Predictor(opt, model, loss, config['lr'], config['epochs'],
                          config['patience'], metrics, loger, device,
                          config['is_collected'])
    # training
    predictor.fit(generator.train_data, generator.val_data)

    # predict
    test_loss, Y, hat_Y = predictor.predict(generator.test_data)
    eval_res = predictor.evaluate(Y, hat_Y)

    print(f'test loss: {test_loss}')

    for m in eval_res:
        print(f'{m[0]}: {m[1]:.5f}')

if __name__ == '__main__':
    run()
