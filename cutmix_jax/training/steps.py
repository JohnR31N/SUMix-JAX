from functools import partial

import jax
import jax.numpy as jnp

from cutmix_jax.cutmix import cutmix_batch
from cutmix_jax.losses import classification_loss, cutmix_loss
from cutmix_jax.sumix import sumix_loss
from cutmix_jax.training.state import get_logits


def _sumix_metrics_dict(loss, logits, info, sumix_info):
    metrics = {
        "loss": loss,
        "acc": jnp.mean(jnp.argmax(logits, axis=-1) == info["labels_a"]),
        "lam": sumix_info["lam"],
        "lam_min": sumix_info["lam_min"],
        "lam_max": sumix_info["lam_max"],
        "lam_std": sumix_info["lam_std"],
        "cls_loss": sumix_info["cls_loss"],
        "reg_loss": sumix_info["reg_loss"],
        "alpha_a_y_mean": sumix_info["alpha_a_y_mean"],
        "alpha_b_y_mean": sumix_info["alpha_b_y_mean"],
        "info_a_y_mean": sumix_info["info_a_y_mean"],
        "info_b_y_mean": sumix_info["info_b_y_mean"],
    }

    # Requires the raw-alpha version of cutmix_jax.sumix.sumix_loss.
    # Keeping these keys here makes the training log expose the pre-scale alpha values.
    if "raw_alpha_a_y_mean" in sumix_info:
        metrics["raw_alpha_a_y_mean"] = sumix_info["raw_alpha_a_y_mean"]
        metrics["raw_alpha_b_y_mean"] = sumix_info["raw_alpha_b_y_mean"]

    return metrics


@jax.jit
def train_step_baseline(state, batch):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        outputs, new_model_state = state.apply_fn(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
        )
        logits = get_logits(outputs)
        loss = classification_loss(logits, labels)
        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return state, {"loss": loss, "acc": acc}


@partial(jax.jit, static_argnames=("cutmix_alpha",))
def train_step_cutmix(state, batch, rng, cutmix_alpha: float):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    mixed_images, info = cutmix_batch(
        images=images,
        labels=labels,
        rng=rng,
        alpha=cutmix_alpha,
    )

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        outputs, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )
        logits = get_logits(outputs)
        loss = cutmix_loss(logits, info)
        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == info["labels_a"])

    return state, {"loss": loss, "acc": acc, "lam": info["lam"]}


@partial(
    jax.jit,
    static_argnames=("cutmix_alpha", "sumix_gamma", "sumix_alpha_scale"),
)
def train_step_cutmix_sumix(
    state,
    batch,
    rng,
    cutmix_alpha: float,
    sumix_gamma: float,
    sumix_alpha_scale: float,
):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    mixed_images, info = cutmix_batch(
        images=images,
        labels=labels,
        rng=rng,
        alpha=cutmix_alpha,
    )

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}

        # Official-alignment: original and mixed branches both use train=True.
        outputs_one, _ = state.apply_fn(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
        )
        cls_one, uncertain_one = outputs_one

        outputs_mix, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )
        cls_mix, uncertain_mix = outputs_mix

        loss, sumix_info = sumix_loss(
            cls_one=cls_one,
            uncertain_one=uncertain_one,
            cls_mix=cls_mix,
            uncertain_mix=uncertain_mix,
            labels_a=info["labels_a"],
            labels_b=info["labels_b"],
            perm=info["perm"],
            lam_area=info["lam"],
            gamma=sumix_gamma,
            alpha_scale=sumix_alpha_scale,
        )
        return loss, (cls_mix, new_model_state, sumix_info)

    (loss, (logits, new_model_state, sumix_info)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])

    return state, _sumix_metrics_dict(loss, logits, info, sumix_info)


@partial(jax.pmap, axis_name="batch")
def train_step_baseline_pmap(state, batch):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        outputs, new_model_state = state.apply_fn(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
        )
        logits = get_logits(outputs)
        loss = classification_loss(logits, labels)
        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    new_batch_stats = jax.lax.pmean(new_model_state["batch_stats"], axis_name="batch")

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    acc = jax.lax.pmean(acc, axis_name="batch")

    return state, {"loss": loss, "acc": acc}


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3,))
def train_step_cutmix_pmap(state, batch, rng, cutmix_alpha: float):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    mixed_images, info = cutmix_batch(
        images=images,
        labels=labels,
        rng=rng,
        alpha=cutmix_alpha,
    )

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        outputs, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )
        logits = get_logits(outputs)
        loss = cutmix_loss(logits, info)
        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    new_batch_stats = jax.lax.pmean(new_model_state["batch_stats"], axis_name="batch")

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == info["labels_a"])
    acc = jax.lax.pmean(acc, axis_name="batch")
    lam = jax.lax.pmean(info["lam"], axis_name="batch")

    return state, {"loss": loss, "acc": acc, "lam": lam}


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5))
def train_step_cutmix_sumix_pmap(
    state,
    batch,
    rng,
    cutmix_alpha: float,
    sumix_gamma: float,
    sumix_alpha_scale: float,
):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    mixed_images, info = cutmix_batch(
        images=images,
        labels=labels,
        rng=rng,
        alpha=cutmix_alpha,
    )

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}

        outputs_one, _ = state.apply_fn(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
        )
        cls_one, uncertain_one = outputs_one

        outputs_mix, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )
        cls_mix, uncertain_mix = outputs_mix

        loss, sumix_info = sumix_loss(
            cls_one=cls_one,
            uncertain_one=uncertain_one,
            cls_mix=cls_mix,
            uncertain_mix=uncertain_mix,
            labels_a=info["labels_a"],
            labels_b=info["labels_b"],
            perm=info["perm"],
            lam_area=info["lam"],
            gamma=sumix_gamma,
            alpha_scale=sumix_alpha_scale,
        )
        return loss, (cls_mix, new_model_state, sumix_info)

    (loss, (logits, new_model_state, sumix_info)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    new_batch_stats = jax.lax.pmean(new_model_state["batch_stats"], axis_name="batch")

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    metrics = _sumix_metrics_dict(loss, logits, info, sumix_info)
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return state, metrics


@jax.jit
def eval_step(state, batch):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    variables = {"params": state.params, "batch_stats": state.batch_stats}
    outputs = state.apply_fn(variables, images, train=False, mutable=False)
    logits = get_logits(outputs)

    loss = classification_loss(logits, labels)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return {"loss": loss, "acc": acc}
