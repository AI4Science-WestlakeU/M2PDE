import torch
from tqdm.auto import tqdm
from torch import nn


def compose_diffusion(
    model_list,
    shape: list,
    update_f: list,
    normalize_f,
    unnormalize_f,
    other_condition=[],
    num_iter=2,
    device="cuda",
):
    """compose diffusion model

    Args:
        model_list (_type_):conditional diffusion model for each physics field
        shape (_type_): shape of field: b, c, *
        update_f (list): update function for each physics field
        normalize_f (_type_, optional): normalization function for each physics field.
        unnormalize_f (_type_, optional): unnormalization function for each physics field.
        other_condition (list): other_condition such as initial state, source term.
        num_iter: (int, optional): outer iteration. Defaults to 2.
        device (str, optional): _description_. Defaults to 'cuda'.
    Returns:
        list: a list contains each field
    """
    with torch.no_grad():

        n_compose = len(model_list)

        timestep = model_list[0].num_timesteps

        # initial field
        mult_p_estimate = []
        for s in shape:
            mult_p_estimate.append(torch.randn(s, device=device))

        for k in range(num_iter):
            mult_p_estimate_before = mult_p_estimate.copy()
            mult_p_estimate = []
            mult_p = []
            for s in shape:
                mult_p_estimate.append(torch.randn(s, device=device))
                mult_p.append(torch.randn(s, device=device))
            for t in tqdm(reversed(range(0, timestep)), desc="sampling loop time step", total=timestep):
                alpha = 1 - t / (timestep - 1) if k > 0 else 1
                for i in range(n_compose):
                    # condition
                    model = model_list[i]
                    update = update_f[i]
                    cond = update(
                        alpha,
                        mult_p_estimate.copy(),
                        mult_p_estimate_before.copy(),
                        other_condition,
                        normalize_f,
                        unnormalize_f,
                    )
                    mult_p[i], x0 = model.p_sample(mult_p[i].clone(), t, cond)
                    # update estimated physics field

                    mult_p_estimate[i] = model.unnormalize(x0)
    return mult_p


def compose_diffusion_ddim(
    model_list,
    shape: list,
    update_f: list,
    normalize_f,
    unnormalize_f,
    other_condition=[],
    num_iter=2,
    device="cuda",
    clip_denoised=True,
):
    """compose diffusion model

    Args:
        model_list (_type_):conditional diffusion model for each physics field
        shape (_type_): shape of field: b, c, *
        update_f (list): update function for each physics field
        normalize_f (_type_, optional): normalization function for each physics field.
        unnormalize_f (_type_, optional): unnormalization function for each physics field.
        other_condition (list): other_condition such as initial state, source term.
        num_iter: (int, optional): outer iteration. Defaults to 2.
        device (str, optional): _description_. Defaults to 'cuda'.
    Returns:
        list: a list contains each field
    """
    with torch.no_grad():

        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0][0],
            model_list[0].betas.device,
            model_list[0].num_timesteps,
            model_list[0].sampling_timesteps,
            model_list[0].ddim_sampling_eta,
            model_list[0].objective,
        )

        n_compose = len(model_list)

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # initial field
        mult_p_estimate = []
        for s in shape:
            mult_p_estimate.append(torch.randn(s, device=device))

        for k in range(num_iter):
            mult_p_estimate_before = mult_p_estimate.copy()
            mult_p_estimate = []
            mult_p = []
            for s in shape:
                mult_p_estimate.append(torch.randn(s, device=device))
                mult_p.append(torch.randn(s, device=device))
            for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
                Lambda = 1 - time_next / (total_timesteps - 1) if k > 0 else 1
                for i in range(n_compose):
                    # condition
                    model = model_list[i]
                    update = update_f[i]
                    cond = update(
                        Lambda,
                        mult_p_estimate.copy(),
                        mult_p_estimate_before.copy(),
                        other_condition,
                        normalize_f,
                        unnormalize_f,
                    )
                    time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                    pred_noise, x_start, *_ = model.model_predictions(
                        mult_p[i].clone(), time_cond, cond, x_self_cond=None, clip_x_start=clip_denoised
                    )
                    if time_next < 0:
                        mult_p[i] = x_start
                        continue

                    alpha = model.alphas_cumprod[time]
                    alpha_next = model.alphas_cumprod[time_next]

                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma**2).sqrt()

                    noise = torch.randn_like(mult_p[i])

                    mult_p[i] = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

                    # update estimated physics field

                    mult_p_estimate[i] = model.unnormalize(x_start)
    return mult_p


def compose_diffusion_multiE(
    model,
    shape,
    cond_shape,
    update_f,
    adj,
    boundary_emb,
    normalize_f=nn.Identity(),
    unnormalize_f=nn.Identity(),
    other_condition=[],
    num_iter=2,
    device="cuda",
):
    """compose diffusion model for multi element.

    Args:
        model: conditional diffusion model.
        shape: shape of field.
        update_f: update function physics field.
        adj (dict): neighbor for each element.
        normalize_f (_type_, optional): normalization function for each physics field.
        unnormalize_f (_type_, optional): unnormalization function for each physics field.
        boundary_emb: emb function for boundary.
        other_condition (list): other_condition such as initial state, source term. The shape of list element is b, *
        unnormalize (_type_, optional): unnormalization function for different physics field. Defaults to identity.
        num_iter: (int, optional): outer iteration. Defaults to 2.
        device (str, optional): _description_. Defaults to 'cuda'.
    Returns:
        Tensor: a tensor of multiphysics field
    """
    with torch.no_grad():

        n_compose = len(adj)

        timestep = model.num_timesteps

        # initial field
        mult_e_estimate = torch.randn((n_compose,) + shape).to(device)
        # for i in range(n_compose):
        #     mult_p_estimate.append(torch.randn(shape, device=device))

        for k in range(num_iter):
            mult_e_estimate_before = mult_e_estimate.clone()
            mult_e_estimate = torch.randn((n_compose,) + shape).to(device)
            mult_e = torch.randn((n_compose,) + shape).to(device)
            for t in tqdm(reversed(range(0, timestep)), desc="sampling loop time step", total=timestep):
                alpha = 1 - t / (timestep - 1) if k > 0 else 1
                cond = update_f(
                    alpha,
                    adj,
                    cond_shape,
                    boundary_emb,
                    mult_e_estimate.clone(),
                    mult_e_estimate_before.clone(),
                    other_condition,
                    normalize_f,
                    unnormalize_f,
                )
                mult_e, x0 = model.p_sample(mult_e.clone(), t, cond)
                mult_e_estimate = model.unnormalize(x0)
    return mult_e


def compose_diffusion_multiE_ddim(
    model,
    shape,
    cond_shape,
    update_f,
    adj,
    boundary_emb,
    normalize_f=nn.Identity(),
    unnormalize_f=nn.Identity(),
    other_condition=[],
    num_iter=2,
    device="cuda",
    clip_denoised=True,
):
    """compose diffusion model for multi element.

    Args:
        model: conditional diffusion model.
        shape: shape of field.
        update_f: update function physics field.
        adj (dict): neighbor for each element.
        normalize_f (_type_, optional): normalization function for each physics field.
        unnormalize_f (_type_, optional): unnormalization function for each physics field.
        boundary_emb: emb function for boundary.
        other_condition (list): other_condition such as initial state, source term. The shape of list element is b, *
        unnormalize (_type_, optional): unnormalization function for different physics field. Defaults to identity.
        num_iter: (int, optional): outer iteration. Defaults to 2.
        device (str, optional): _description_. Defaults to 'cuda'.
    Returns:
        Tensor: a tensor of multiphysics field
    """
    with torch.no_grad():

        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            len(adj),
            model.betas.device,
            model.num_timesteps,
            model.sampling_timesteps,
            model.ddim_sampling_eta,
            model.objective,
        )

        n_compose = len(adj)

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # initial field
        mult_e_estimate = torch.randn((n_compose,) + shape).to(device)

        for k in range(num_iter):
            mult_e_estimate_before = mult_e_estimate.clone()
            mult_e_estimate = torch.randn((n_compose,) + shape).to(device)
            mult_e = torch.randn((n_compose,) + shape).to(device)
            for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
                Lambda = 1 - time_next / (total_timesteps - 1) if k > 0 else 1
                cond = update_f(
                    Lambda,
                    adj,
                    cond_shape,
                    boundary_emb,
                    mult_e_estimate.clone(),
                    mult_e_estimate_before.clone(),
                    other_condition,
                    normalize_f,
                    unnormalize_f,
                )
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                pred_noise, x_start, *_ = model.model_predictions(
                    mult_e, time_cond, cond, x_self_cond=None, clip_x_start=clip_denoised
                )
                if time_next < 0:
                    mult_e = x_start
                    continue

                alpha = model.alphas_cumprod[time]
                alpha_next = model.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma**2).sqrt()

                noise = torch.randn_like(mult_e)

                mult_e = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

                # update estimated physics field

                mult_e_estimate = model.unnormalize(x_start)
    return mult_e
