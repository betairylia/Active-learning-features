from jax import random
from neural_tangents import stax
import jax.numpy as jnp
import neural_tangents as nt

from matplotlib import pyplot as plt

from datamodules import get_train_X, get_val_X, get_Y

hidden_dim = 2048
in_dim = 1

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(hidden_dim, W_std=1.5, b_std=0.05), stax.Erf(),# stax.Erf(),
    stax.Dense(hidden_dim, W_std=1.5, b_std=0.05), stax.Erf(),# stax.Erf(),
    # stax.Dense(hidden_dim, W_std=1.5, b_std=0.05), stax.Erf(),# stax.Erf(),
    # stax.Dense(hidden_dim, W_std=1.5, b_std=0.05), stax.Erf(),# stax.Erf(),
    # stax.Dense(hidden_dim, W_std=1.5, b_std=0.05), stax.Erf(),# stax.Erf(),
    stax.Dense(1)
)

key = random.PRNGKey(1)
# x = random.normal(key, (10, 1))

# Data generation
ntx = get_train_X(N = 3)
nvx = get_val_X(N = 128)

x_tr = jnp.array(ntx)[:, None]
x_va = jnp.array(nvx)[:, None]
y_tr = jnp.array(get_Y(ntx, 0.03))[:, None]
y_va = jnp.array(get_Y(nvx))[:, None]

_, params = init_fn(key, input_shape=x_tr.shape)

################### Direct computation of Linearized dynamics on MSE #####################

predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_tr, y_tr, diag_reg = 7e-4)
# y_val_nngp = predict_fn(x_test = x_va, get = 'nngp')

y_val_ntk, y_val_ntk_cov = predict_fn(x_test = x_va, get = 'ntk', compute_cov = True)
# y_val_ntk = predict_fn(x_test = x_va, get = 'nngp', compute_cov = True)

y = apply_fn(params, x_va)
y_var = kernel_fn(x_va, x_va, 'nngp')

print(x_va)
print(y_val_ntk)

ntk_mean = jnp.reshape(y_val_ntk, (-1,))
ntk_std = jnp.sqrt(jnp.diag(y_val_ntk_cov))

cov_scale = 2
y_lower = ntk_mean - ntk_std * cov_scale
y_upper = ntk_mean + ntk_std * cov_scale

##################### Upper bound by Gronwall Inequality ######################

Ozz = jnp.diag(kernel_fn(x_va, x_va, 'ntk'))
Oxx = jnp.diag(kernel_fn(x_tr, x_tr, 'ntk'))
Ozx = kernel_fn(x_va, x_tr, 'ntk')

addition_term = jnp.min(Oxx[None, :] - 2 * Ozx, axis = 1)
UB = Ozz + addition_term

UB = 0.2 * jnp.exp(2 * jnp.sqrt(UB) - 1)

mean_term = jnp.mean(Oxx[None, :] - 2 * Ozx, axis = 1)
UB_mean = Ozz + mean_term

############################### VISUALIZATION #################################

plt.figure(figsize = (8, 6))
plt.tight_layout()
# plt.figure(figsize = (14, 10))

# plt.fill_between(x_va[:, 0], ntk_mean - Ozz, ntk_mean + Ozz, alpha = 0.04, color = 'C3', label = "O(z,z)")
# plt.fill_between(x_va[:, 0], ntk_mean - UB_mean, ntk_mean + UB_mean, alpha = 0.14, color = 'C0', label = "Upper bound (mean)")
plt.fill_between(x_va[:, 0], ntk_mean - UB, ntk_mean + UB, alpha = 0.14, color = 'C2', label = "Upper bound (inf)")
plt.fill_between(x_va[:, 0], y_lower, y_upper, alpha = 0.3, color = 'C1', label = "Exact trained ensemble")

plt.plot(x_va[:, 0], y_va[:, 0], label = "target")
plt.plot(x_va[:, 0], ntk_mean, label = "trained")
plt.plot(x_tr[:, 0], y_tr[:, 0], 'o', label = "dataset")

plt.legend()

plt.savefig("/mnt/out.png")

