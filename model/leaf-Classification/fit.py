import mxnet as mx
import logging
import os
import time
import re

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str, default="30,60",
                       help='the epochs to reduce the lr, [default = 30,60]')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='load model prefix')
    train.add_argument('--save-prefix', type=str,
                       help='save model prefix')
    train.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--resume', dest='resume', type=bool, default=0,
                       help='resume training from epoch n [default = 0]')
    train.add_argument('--finetune', dest='finetune', type=bool, default=0,
                       help='finetune from epoch n, rename the model before doing this [default = 0]')
    train.add_argument('--pretrained', dest='pretrained', help='pretrained from epoch n [default = 0]',
                       default=0, type=bool)
    return train

def convert_pretrained(name, sym, arg_params, args):
    if 'mobilenet' in name:
        internals = sym.get_internals()
        pool6 = internals['pool6_output']
        fc7 = mx.symbol.Convolution(data=pool6, no_bias=False, num_filter=args.num_classes, kernel=(1,1), name='fc7')
        flatten = mx.symbol.Flatten(data=fc7)
        softmax = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')

        re_prog = re.compile('fc7_.*')
        delete_param_names = [name for name in sym.list_arguments() if re_prog.match(name)]
        for name in delete_param_names:
            del arg_params[name]

    return softmax, arg_params
def _get_lr_scheduler(args, num_examples, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    if args.save_prefix is None:
        return None
    dst_dir = os.path.dirname(args.save_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.save_prefix if rank == 0 else "%s-%d" % (
        args.save_prefix, rank))

def learning_rate(lr_scheduler, base_lr=None, frequent=50):
    def _callback(param):
        count = param.nbatch
        if count % frequent == 0:
            if lr_scheduler == None:
                logging.info('Epoch[%d] Batch [%d]\tLearningRate: %s',
                        param.epoch, count, str(base_lr))
            else:
                logging.info('Epoch[%d] Batch [%d]\tLearningRate: %f',
                        param.epoch, count, lr_scheduler.base_lr)
    return _callback

def fit(args, train_iter, val_iter, **kwargs):
    network = None
    num_examples = train_iter.max_iter
    kv = mx.kvstore.create(args.kv_store)

    head = '%(asctime)-15s %(levelname)s %(filename)s %(funcName)s %(lineno)d ' \
            'Node[' + str(kv.rank) + ']\t %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    if args.resume:
        assert(not args.finetune and not args.pretrained)
        logger.info("Resume training from resume {}"
                .format(args.resume))
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        begin_epoch = args.load_epoch
    elif args.finetune:
        assert(not args.resume and not args.pretrained)
        logging.info("Start finetuning from finetune {}"
            .format(args.finetune))
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        network = sym
        begin_epoch = args.load_epoch
    elif args.pretrained:
        assert(not args.resume and not args.finetune)
        logging.info("Start pretraining with model prefix {}, epoch {}"
            .format(args.model_prefix, args.load_epoch))
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        sym, arg_params = convert_pretrained(args.model_prefix, sym, arg_params, args)
        network = sym
        begin_epoch = 0
    else:
        logging.info("Experimental: start training from scratch")
        arg_params = None
        aux_params = None
        begin_epoch = 0

    if network == None:
        from importlib import import_module
        net = import_module('symbols.'+args.network)
        network = net.get_symbol(**vars(args))

    # checkpoint callback
    checkpoint = _save_model(args, kv.rank)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, num_examples, kv)

    # create model
    model = mx.mod.Module(
        data_names    = train_iter.input_names,
        label_names   = train_iter.label_names,
        context       = devs,
        symbol        = network
    )

    lr_scheduler  = lr_scheduler
    optimizer_params = {
            'learning_rate': lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

    if args.network == 'alexnet':
        # AlexNet will not converge using Xavier
        initializer = mx.init.Normal()
    else:
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    # evaluation metrices
    eval_metrics = ['accuracy', 'ce']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = [learning_rate(lr_scheduler, frequent=args.disp_batches, base_lr=lr),
                           mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train_iter,
        begin_epoch        = begin_epoch,
        num_epoch          = args.num_epochs,
        eval_data          = val_iter,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)
