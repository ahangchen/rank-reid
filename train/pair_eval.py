from keras.engine import Model
from keras.models import load_model

from baseline.evaluate import train_predict, test_predict


def train_pair_eval():
    model = load_model('pair_pretrain.h5')
    model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
                  outputs=[model.get_layer('model_1').get_output_at(0)])
    train_predict(model, '.')


def test_pair_eval():
    model = load_model('pair_pretrain.h5')
    model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
                  outputs=[model.get_layer('model_1').get_output_at(0)])
    test_predict(model, '.')

if __name__ == '__main__':
    test_pair_eval()