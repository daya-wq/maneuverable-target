import torch


def as_body_linear_drag_coeff(
    value,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return body-frame linear drag coefficients as a 3-vector in 1/s.

    A scalar input is expanded isotropically. A length-3 input maps to
    coefficients along the body x/y/z axes respectively.
    """

    coeff = torch.as_tensor(value, device=device, dtype=dtype)
    if coeff.numel() == 1:
        coeff = coeff.reshape(1).repeat(3)
    elif coeff.ndim >= 1 and coeff.shape[-1] == 3:
        return coeff
    elif coeff.numel() == 3:
        coeff = coeff.reshape(3)
    else:
        raise ValueError(
            "Linear drag coefficients must be a scalar or a length-3 sequence."
        )
    return coeff


def compute_body_linear_drag_force(
    vel_body: torch.Tensor,
    drag_coeff: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    """Compute body-frame linear drag force.

    Args:
        vel_body: Body-frame linear velocity in m/s.
        drag_coeff: Mass-normalized body-frame linear drag coefficients in 1/s.
        mass: Vehicle mass in kg, broadcastable to vel_body[..., :1].
    """

    coeff = as_body_linear_drag_coeff(
        drag_coeff,
        device=vel_body.device,
        dtype=vel_body.dtype,
    )
    if coeff.ndim == 1:
        coeff = coeff.reshape(*([1] * (vel_body.ndim - 1)), 3)
    mass = torch.as_tensor(mass, device=vel_body.device, dtype=vel_body.dtype)
    return -(mass * coeff) * vel_body


def as_world_vertical_aero_coeff(
    value,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return a scalar world-vertical aerodynamic coefficient in 1/s."""

    coeff = torch.as_tensor(value, device=device, dtype=dtype)
    if coeff.numel() == 1:
        return coeff.reshape(())
    if coeff.ndim >= 1 and coeff.shape[-1] == 1:
        return coeff
    raise ValueError(
        "World-vertical aerodynamic coefficient must be a scalar or end with a singleton dimension."
    )


def compute_world_vertical_aero_force(
    vel_world: torch.Tensor,
    vertical_coeff: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    """Compute a world-frame vertical aerodynamic correction.

    This term is used for the literature-identified Crazyflie vertical net
    aerodynamic effect. It depends only on world vertical velocity so that
    large attitude changes do not convert horizontal speed into a spurious
    body-z force channel.
    """

    coeff = as_world_vertical_aero_coeff(
        vertical_coeff,
        device=vel_world.device,
        dtype=vel_world.dtype,
    )
    mass = torch.as_tensor(mass, device=vel_world.device, dtype=vel_world.dtype)
    force = torch.zeros_like(vel_world)
    force[..., 2:3] = (mass * coeff) * vel_world[..., 2:3]
    return force
