import argparse
import pathlib
import pickle

import tensorflow as tf
from dnnlib.tflib.network import Network
from dnnlib.util import EasyDict


def convert(
    network_pkl: pathlib.Path, save_dir: pathlib.Path, res: int, truncation_psi: float,
) -> None:
    with tf.compat.v1.Session() as sess:
        spec = EasyDict(map=2, fmaps=1 if res >= 512 else 0.5)
        G_args = EasyDict(
            func_name="training.networks.G_main",
            fmap_base=int(spec.fmaps * 16384),
            mapping_layers=spec.map,
        )
        G = Network(
            "G",
            num_channels=3,
            resolution=res,
            use_noise=False,
            return_dlatents=True,
            truncation_psi=truncation_psi,
            **G_args,
        )
        Gs = G.clone("Gs")
        with open(str(network_pkl.resolve()), "rb") as fp:
            _, _, rGs = pickle.load(fp)
        Gs.copy_vars_from(rGs)

        input_names = [t.name for t in Gs.input_templates]
        output_names = [t.name for t in Gs.output_templates]
        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), [t.op.name for t in Gs.output_templates]
        )

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        inputs = [graph.get_tensor_by_name(input_names[0])]
        outputs = [graph.get_tensor_by_name(name) for name in output_names]
        images = tf.transpose(outputs[0], [0, 2, 3, 1])
        images = tf.saturate_cast((images + 1.0) * 127.5, tf.uint8)

        builder = tf.compat.v1.saved_model.Builder(str(save_dir.resolve()))
        default = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        signature_def_map = {
            default: tf.compat.v1.saved_model.build_signature_def(
                {"latents": tf.saved_model.utils.build_tensor_info(inputs[0])},
                {"images": tf.saved_model.utils.build_tensor_info(images)},
            ),
            "mapping": tf.compat.v1.saved_model.build_signature_def(
                {"latents": tf.saved_model.utils.build_tensor_info(inputs[0])},
                {"dlatents": tf.saved_model.utils.build_tensor_info(outputs[1])},
            ),
            "synthesis": tf.compat.v1.saved_model.build_signature_def(
                {"dlatents": tf.saved_model.utils.build_tensor_info(outputs[1])},
                {"images": tf.saved_model.utils.build_tensor_info(images)},
            ),
        }
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING], signature_def_map
        )
        builder.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network_pkl", type=pathlib.Path)
    parser.add_argument("save_dir", type=pathlib.Path)
    parser.add_argument(
        "--res",
        help="Dataset resolution (default: 256)",
        type=int,
        metavar="INT",
        default=256,
    )
    parser.add_argument(
        "--trunc",
        dest="truncation_psi",
        type=float,
        help="Truncation psi (default: %(default)s)",
        default=0.5,
    )
    args = parser.parse_args()

    if args.save_dir.exists():
        raise FileExistsError(args.save_dir.resolve())
    else:
        convert(**vars(args))
