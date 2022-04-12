from dis import code_info
import yaml

from torch.optim import Adam
from torch.nn import L1Loss
from models.dcrnn import DCRNN
from utils.inits import seed_init, device_init
from utils.predictor import Predictor
from utils.supervisor import Generator, Incidentor
from utils.evaluator import mae, rmse, mape
from utils.loger import Loger


def run():
    with open('./config/DCRNN_METR.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file.read(), yaml.SafeLoader)

    seed_init(config['seed'])

    loger = Loger(config['model_name'], config['log_path'])
    device = device_init()

    #data
    incidentor = Incidentor(config['adj_path'], loger, device)
    incidentor.dual_random_walk()
    adj = incidentor.adj
    supports = incidentor.supports

    generator = Generator(config['data_path'], loger, config['windows'], config['lag'],
                          config['horizon'], config['train_ratio'], config['val_ratio'],
                          config['bs'], config['is_scaler'], device)
    
    # model and for training
    opt = Adam
    loss = L1Loss()
    model = DCRNN(adj, supports, config['windows'], config['horizon'], config['d_in'], 
                  config['d_out'], config['max_diff_step'], config['cl_decay_steps'],
                  incidentor.nodes, config['layers'], config['units'], config['use_curriculum_learning'])
    
    metrics = [mae, rmse, mape]

    predictor = Predictor(opt, model, loss, config['lr'], config['epochs'],
                          config['patience'], metrics, loger, device,
                          config['is_collected'], config['scheduled_sampling'])
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
