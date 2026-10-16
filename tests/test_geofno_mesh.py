"""Mesh-native Geo-FNO (methods.geofno.model_deform): the point-cloud Fourier layers
must be the same operators as the grid ones, so a point cloud lying on the latent grid
reproduces rfft2/irfft2, and padded sensor slots must not reach the prediction."""
import torch

from methods.geofno.model_deform import GeoFNOSupervisedMesh, SpectralConv2d


def _grid_points(s1, s2, batch):
    gx = torch.arange(s1).float() / s1
    gy = torch.arange(s2).float() / s2
    return torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(1, -1, 2).expand(batch, -1, -1)


def test_point_layers_match_grid_layers_on_the_latent_grid():
    torch.manual_seed(0)
    s1, s2, channels = 20, 12, 3
    conv = SpectralConv2d(channels, channels, 6, 5, s1, s2)
    u = torch.randn(2, channels, s1, s2)
    pts = _grid_points(s1, s2, 2)
    ref = conv(u)
    weights = torch.full((2, s1 * s2), 1.0 / (s1 * s2))
    assert torch.allclose(conv(u.flatten(-2), xi_in=pts, w_in=weights), ref, atol=1e-5)
    assert torch.allclose(conv(u, xi_out=pts).view_as(ref), ref, atol=1e-5)
    # one query mesh shared by the batch
    assert torch.allclose(conv(u, xi_out=pts[:1]).view_as(ref), ref, atol=1e-5)


def test_padded_sensors_do_not_reach_the_prediction():
    torch.manual_seed(0)
    model = GeoFNOSupervisedMesh(n_fields=3, latent_Nx=16, latent_Ny=16, n_modes_x=4, n_modes_y=4,
                                 hidden_channels=8, n_layers=1, iphi_width=8).eval()
    for p in model.parameters():                     # move off the identity-deformation init
        torch.nn.init.normal_(p, std=0.1) if p.is_floating_point() else None
    batch, m, n = 2, 12, 50
    coords = torch.rand(1, n, 2)
    obs_coords, obs_values = torch.rand(batch, m, 2), torch.randn(batch, m, 1)
    obs_mask = torch.ones(batch, m)
    field_ids = torch.arange(m).remainder(2).expand(batch, m).clone()
    obs_mask[1, 8:] = 0
    field_ids[1, 8:] = -1
    with torch.no_grad():
        ref = model(coords, obs_coords, obs_values, obs_mask, field_ids)
        obs_values[1, 8:] = 99.0
        obs_coords[1, 8:] = 0.123
        out = model(coords, obs_coords, obs_values, obs_mask, field_ids)
    assert ref.shape == (batch, n, 3)
    assert torch.equal(out, ref)
