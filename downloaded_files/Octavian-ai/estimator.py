
import tensorflow as tf

from .model import model_fn

def get_estimator(args):

	run_config = tf.estimator.RunConfig(
		model_dir=args["model_dir"],
		tf_random_seed=args["random_seed"],
		save_checkpoints_steps=None,
		save_checkpoints_secs=args["eval_every"],
	)

	return tf.estimator.Estimator(
		model_fn=model_fn, 
		config=run_config,
		warm_start_from=args["warm_start_dir"],
		params=args)