from argparse import ArgumentParser
import os

def get_args():
    parser = ArgumentParser(description="SM CNN")

    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--dataset', type=str, help='trecqa|wikiqa', default='trecqa')
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--word-vectors-dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'GloVe'))
    parser.add_argument('--word-vectors-file', help='word vectors filename', default='glove.840B.300d.txt')
    parser.add_argument('--word-vectors-dim', type=int, default=300,
                        help='number of dimensions of word vectors (default: 300)')
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--dev_every', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--output_channel', type=int, default=100)
    parser.add_argument('--filter_width', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--vector_cache', type=str, default='data/word2vec.trecqa.pt')
    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--onnx', action='store_true', help='export model to onnx')

    # added so that no compile error
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status (default: 10)')

    # parser.add_argument('--epochs', type=int, default=30)
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--mode', type=str, default='static')
    # parser.add_argument('--lr', type=float, default=1.0)
    # parser.add_argument('--seed', type=int, default=3435)
    # parser.add_argument('--dataset', type=str, help='trecqa|wikiqa', default='trecqa')
    # parser.add_argument('--resume_snapshot', type=str, default=None)
    # parser.add_argument('--dev_every', type=int, default=30)
    # parser.add_argument('--log_every', type=int, default=10)
    # parser.add_argument('--patience', type=int, default=50)
    # parser.add_argument('--save_path', type=str,    default='saves')
    # parser.add_argument('--output_channel', type=int, default=100)
    # parser.add_argument('--filter_width', type=int, default=5)
    # parser.add_argument('--words_dim', type=int, default=50)
    # parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--epoch_decay', type=int, default=15)
    # parser.add_argument('--vector_cache', type=str, default='data/word2vec.trecqa.pt')
    # parser.add_argument('--trained_model', type=str, default="")
    # parser.add_argument('--weight_decay',type=float, default=1e-5)
    # parser.add_argument('--onnx', action='store_true', help='export model to onnx')
    # parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    # parser.add_argument('--word-vectors-dir', help='word vectors directory',
    #                     default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'GloVe'))
    # parser.add_argument('--word-vectors-file', help='word vectors filename', default='glove.840B.300d.txt')
    # parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    #
    #
    # parser.add_argument('model_outfile', help='file to save final model')
    # parser.add_argument('--arch', help='model architecture to use', choices=['mpcnn', 'mpcnn_lite'], default='mpcnn')
    # parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid, trecqa, wikiqa]', default='sick')
    # parser.add_argument('--word-vectors-dir', help='word vectors directory',
    #                     default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'GloVe'))
    # parser.add_argument('--word-vectors-file', help='word vectors filename', default='glove.840B.300d.txt')
    # parser.add_argument('--word-vectors-dim', type=int, default=300,
    #                     help='number of dimensions of word vectors (default: 300)')
    # parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    # parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    # parser.add_argument('--wide-conv', action='store_true', default=False,
    #                     help='use wide convolution instead of narrow convolution (default: false)')
    # parser.add_argument('--attention', choices=['none', 'basic', 'idf'], default='none', help='type of attention to use')
    # parser.add_argument('--sparse-features', action='store_true',
    #                     default=False, help='use sparse features (default: false)')
    # parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    # parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use: adam or sgd (default: adam)')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    # parser.add_argument('--lr-reduce-factor', type=float, default=0.3,
    #                     help='learning rate reduce factor after plateau (default: 0.3)')
    # parser.add_argument('--patience', type=float, default=2,
    #                     help='learning rate patience after seeing plateau (default: 2)')
    # parser.add_argument('--momentum', type=float, default=0, help='momentum (default: 0)')
    # parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimizer epsilon (default: 1e-8)')
    # parser.add_argument('--log-interval', type=int, default=10,
    #                     help='how many batches to wait before logging training status (default: 10)')
    # parser.add_argument('--regularization', type=float, default=0.0001,
    #                     help='Regularization for the optimizer (default: 0.0001)')
    # parser.add_argument('--max-window-size', type=int, default=3,
    #                     help='windows sizes will be [1,max_window_size] and infinity (default: 3)')
    # parser.add_argument('--holistic-filters', type=int, default=300, help='number of holistic filters (default: 300)')
    # parser.add_argument('--per-dim-filters', type=int, default=20, help='number of per-dimension filters (default: 20)')
    # parser.add_argument('--hidden-units', type=int, default=150,
    #                     help='number of hidden units in each of the two hidden layers (default: 150)')
    # parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability (default: 0.5)')
    # parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
    # parser.add_argument('--tensorboard', action='store_true', default=False,
    #                     help='use TensorBoard to visualize training (default: false)')
    # parser.add_argument('--run-label', type=str, help='label to describe run')
    # parser.add_argument('--keep-results', action='store_true',
    #                     help='store the output score and qrel files into disk for the test set')

    args = parser.parse_args()
    return args
