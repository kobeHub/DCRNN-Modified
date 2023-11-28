# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import tensorflow as tf
import yaml

from lib.utils import load_graph_struct, load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        if graph_pkl_filename.endswith(".pkl"):
            _,_, adj_mx = load_graph_data(graph_pkl_filename)
        else:
            adj_mx = load_graph_struct(graph_pkl_filename)

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    tf.keras.backend.set_floatx('float64')
    main(args)
