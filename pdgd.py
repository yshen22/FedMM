from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class PDGradientDescent(optimizer.Optimizer):
	"""Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""

	def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
		super(PDGradientDescent, self).__init__(use_locking, name)
		self._lr = learning_rate
		self._mu = mu


		# Tensor versions of the constructor arguments, created in _prepare().
		self._lr_t = None
		self._mu_t = None

	def _prepare(self):
		self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
		self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

	def _create_slots(self, var_list):
		# Create slots for the global solution.
		for v in var_list:
			self._zeros_slot(v, "vstar", self._name)
			self._zeros_slot(v, 'lambda', self._name)
		self._create_non_slot_variable(1., 'learning_decay_rate',  v[0])

	def _apply_dense(self, grad, var):
		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
		mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
		vstar = self.get_slot(var, "vstar")
		lambda_i = self.get_slot(var, "lambda")
		learning_decay = self._get_non_slot_variable("learning_decay_rate", var.graph)
#		print(learning_decay)
		lr_t = lr_t * learning_decay
#		grad1 = grad + mu_t * (var - vstar) + lambda_i
#		grad1 = tf.minimum(grad1 , self.clip_value * tf.ones_like(grad1))

		var_update = state_ops.assign_sub(var, lr_t * (grad + mu_t * (var - vstar) + lambda_i))

		return control_flow_ops.group(*[var_update, ])

	def _apply_sparse_shared(self, grad, var, indices, scatter_add):
	
		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
		mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
		vstar = self.get_slot(var, "vstar")
		lambda_i = self.get_slot(var, "lambda")
		learning_decay = self._get_non_slot_variable("learning_decay_rate", var.graph)
#		print(learning_decay)
		lr_t = lr_t * learning_decay
	
		v_diff = state_ops.assign(vstar, mu_t * (var - vstar) + lambda_i, use_locking=self._use_locking)
	
		with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
			scaled_grad = scatter_add(vstar, indices, grad)
		var_update = state_ops.assign_sub(var, lr_t * scaled_grad)
	
		return control_flow_ops.group(*[var_update,])

	def _apply_sparse(self, grad, var):
		# lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
		# mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
		# vstar = self.get_slot(var, "vstar")
		# lambda_i = self.get_slot(var, "lambda")
		# indices = grad.indices
		# v_diff = state_ops.assign(vstar, mu_t * (var - vstar) + lambda_i, use_locking=self._use_locking)
		# with ops.control_dependencies([v_diff]):
		# 	scaled_grad = state_ops.scatter_add(vstar, indices, grad)
		# var_update = state_ops.assign_sub(var, lr_t * scaled_grad)

		# return control_flow_ops.group(*[var_update,])

#		return self._apply_dense(grad, var)
#		raise NotImplementedError("Sparse gradient updates are not supported.")

		return self._apply_sparse_shared(
		grad.values, var, grad.indices,
		lambda x, i, v: state_ops.scatter_add(x, i, v))

	def set_params(self, cog, sess):
		all_vars = tf.trainable_variables()
		for variable, value in zip(all_vars, cog):
#			print(variable is None)
#			print(variable.name)
			vstar = self.get_slot(variable, "vstar")
#			print(vstar.name)
			vstar.load(value, sess)

	def set_dual_params(self, cog, sess):
		all_vars = tf.trainable_variables()
		for variable, value in zip(all_vars, cog):
			lambda_i = self.get_slot(variable, "lambda")
			lambda_i.load(value, sess)

	def set_learning_decay(self, learning_decay, graph, sess):
#		self.learning_decay = learning_decay
#		with client.graph.as_default():
		learning_decay_tensor = self._get_non_slot_variable("learning_decay_rate", graph)
		learning_decay_tensor.load(learning_decay, sess)

	@property
	def mu(self):
		return self._mu