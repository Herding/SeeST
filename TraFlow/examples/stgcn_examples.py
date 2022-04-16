import yaml

from torch.optim import Adam
from torch.nn import L1Loss
from models.stgcn import STGCN
from utils.inits import seed_init, device_init
from utils.predictor import Predictor
from utils.supervisor import Generator, Incidentor
from utils.evaluator import mae, rmse, mape
from utils.loger import Loger


def run():
    with open('./config/STGCN_METR.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file.read(), yaml.SafeLoader)

    seed_init(config['seed'])

    loger = Loger(config['model_name'], config['log_path'])
    device = device_init()

    #data
    incidentor = Incidentor(config['adj_path'], loger, device)
    incidentor.cheb_poly_approx(config['order_of_cheb'])

    generator = Generator(config['data_path'], loger, config['windows'], config['lag'],
                          config['horizon'], config['train_ratio'], config['val_ratio'],
                          config['bs'], config['is_scaler'], device)
    
    # model and for training
    opt = Adam
    loss = L1Loss()
    print(incidentor.adj.size())
    model = STGCN(config['windows'], config['horizon'], incidentor.nodes, config['ks'],
                config['kt'], config['blocks'], config['droprate'])
    metrics = [mae, rmse, mape]

    predictor = Predictor(opt, model, loss, config['lr'], config['epochs'],
                          config['patience'], metrics, loger, device,
                          config['is_collected'])
    # training
    predictor.fit(generator.train_data, generator.val_data, incidentor.adj)

    # predict
    test_loss, Y, hat_Y = predictor.predict(generator.test_data, incidentor.adj)
    eval_res = predictor.evaluate(Y, hat_Y)

    print(f'test loss: {test_loss}')

    for m in eval_res:
        print(f'{m[0]}: {m[1]:.5f}')

if __name__ == '__main__':
    run()
