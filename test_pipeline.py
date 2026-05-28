"""
Smoke test for the multilevel bilevel DFO pipeline.
Tests all new components: FISTA, CT, PDE, Tikhonov.
"""
import torch
import numpy as np
import sys
import traceback

torch.manual_seed(42)
np.random.seed(42)

device = "cpu"  # Use CPU for testing

PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [OK] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# Test 1: FISTA solver
def test_fista_solver():
    from FISTA import fista

    # Simple quadratic: min 0.5 * ||Ax - b||^2
    A = torch.eye(10)
    b = torch.randn(10)
    x0 = torch.zeros(10)

    def f(x, y):
        return 0.5 * torch.sum((A @ x - y) ** 2)

    def nabla_f(x, y):
        return A.T @ (A @ x - y)

    def prox_g(x, step):
        return x  # No regularization

    x, L, iters = fista(x0, b, f, nabla_f, prox_g, max_iter=100, L_init=1.0, tol=1e-6)
    residual = torch.norm(x - b)
    assert residual < 1e-4, f"FISTA did not converge: residual={residual:.2e}"


def test_reconstruct_fista():
    from FISTA import reconstruct_fista
    from deepinv.physics import Denoising, GaussianNoise
    from deepinv.optim import L2
    from Regulariser.TV import SmoothedTV

    physics = Denoising(noise_model=GaussianNoise(sigma=0.1), device=device).to(device)
    physics.A_dagger = physics.A_adjoint
    data_fidelity = L2(sigma=1.0)
    regularizer = SmoothedTV(tv_weight=0.01, smoothing=0.01, device=device).to(device)

    gt = torch.rand(1, 1, 32, 32, device=device)
    y = physics(gt)

    x_hat, iters = reconstruct_fista(
        y=y, physics=physics, data_fidelity=data_fidelity,
        regularizer=regularizer, lamda=1.0, step_size=0.1,
        max_iter=50, tol=1e-4,
    )
    assert x_hat.shape == gt.shape, f"Shape mismatch: {x_hat.shape} vs {gt.shape}"


# Test 2: CT physics
def test_ct_physics():
    from deepinv.physics import Tomography, GaussianNoise

    angles = torch.linspace(0, 180, steps=7, device=device)[:-1]
    physics = Tomography(angles=angles, img_width=32, circle=False, normalize=True,
                         noise_model=GaussianNoise(sigma=0.01), device=device).to(device)

    x = torch.rand(1, 1, 32, 32, device=device)
    y = physics(x)
    x_adj = physics.A_adjoint(y)

    assert y.dim() == 4, f"Expected 4D sinogram, got {y.dim()}D"
    assert x_adj.shape == x.shape, f"Adjoint shape mismatch: {x_adj.shape} vs {x.shape}"


# Test 3: Phantom generation
def test_phantom_triangle():
    from problems.phantoms import random_triangle
    img = random_triangle(64, num_triangles=1, device=device)
    assert img.shape == (1, 1, 64, 64), f"Shape mismatch: {img.shape}"
    assert img.min() >= 0.0, f"Negative values in phantom"


def test_phantom_dataset():
    from problems.phantoms import PhantomDataset
    ds = PhantomDataset(n=32, length=5, phantom_type="triangle")
    assert len(ds) == 5
    sample = ds[0]
    assert sample.shape == (1, 32, 32), f"Shape mismatch: {sample.shape}"


# Test 4: CT experiment setup
def test_ct_experiment_setup():
    from problems.ct_radon import setup_ct_experiment
    physics, df, train_ds, val_ds, config = setup_ct_experiment(img_size=32, num_angles=4, device=device)
    assert config["problem"] == "CT"
    sample = train_ds[0]
    y = physics(sample.unsqueeze(0).to(device))
    assert y.dim() == 4


def test_ct_evaluate_objective():
    from problems.ct_radon import setup_ct_experiment
    from DFO import MultilevelSmoothingOptimizer
    from Regulariser.TV import SmoothedTV

    img_size = 32
    physics, df, train_ds, val_ds, config = setup_ct_experiment(img_size=img_size, num_angles=4, device=device)
    
    # Create regularizer
    reg = SmoothedTV(tv_weight=0.01, smoothing=0.01, device=device).to(device)
    
    # Sample gt
    gt = train_ds[0].unsqueeze(0).to(device) # (1, 1, 32, 32)
    
    # Create optimizer
    opt = MultilevelSmoothingOptimizer(
        model=reg,
        physics=physics,
        data_fidelity=df,
        ground_truth=gt,
        lmbd=1.0,
        lr=0.01,
        lower_tol=0.1,
        device=device,
        lower_alg="GD",
        lower_lr=0.1,
    )
    
    # Test evaluate_objective with x_init=None
    # This should internally use A_dagger(y) which is (1, 1, 32, 32)
    # If it uses y.clone() which is (1, 1, 4, 32), it will crash.
    mse, x_hat, iters = opt.evaluate_objective(x_init=None, max_iter=2)
    
    assert x_hat.shape == gt.shape, f"Shape mismatch: {x_hat.shape} vs {gt.shape}"
    assert mse > 0


def test_ct_smoothed_tv():
    from problems.ct_radon import CTSmoothedTV
    reg = CTSmoothedTV(img_size=32, device=device).to(device)
    x = torch.rand(2, 1, 32, 32, device=device)
    energy = reg.g(x)
    grad = reg.grad(x)
    assert energy.shape == (2,), f"Energy shape: {energy.shape}"
    assert grad.shape == x.shape, f"Grad shape: {grad.shape}"


# Test 5: PDE forward model
def test_pde_laplacian():
    from problems.pde_elliptic import create_laplacian_2d
    A = create_laplacian_2d(9)  # 8x8 interior = 64 unknowns
    assert A.shape == (64, 64), f"Laplacian shape: {A.shape}"
    # Should be symmetric positive definite
    A_dense = A.toarray()
    assert np.allclose(A_dense, A_dense.T, atol=1e-10), "Laplacian is not symmetric"


def test_pde_forward_model():
    from problems.pde_elliptic import PDEForwardModel
    model = PDEForwardModel(I_grid=9, K=5, gamma=0.001, device=device)
    u = model.sample_ground_truth(batch_size=3)
    assert u.shape == (3, 64), f"Ground truth shape: {u.shape}"
    d = model.generate_data(u)
    assert d.shape == (3, 5), f"Data shape: {d.shape}"


def test_pde_solve():
    from problems.pde_elliptic import PDEForwardModel
    model = PDEForwardModel(I_grid=9, K=5, gamma=0.001, device=device)
    u = model.sample_ground_truth(batch_size=2)
    d = model.generate_data(u)
    obs_indices = model.initial_obs_indices
    x_hat = model.solve_lower_level(torch.tensor(0.0), obs_indices, d)
    assert x_hat.shape == (2, 64), f"Reconstruction shape: {x_hat.shape}"


def test_pde_bilevel():
    from problems.pde_elliptic import setup_pde_experiment
    pde_model, bilevel_model, config = setup_pde_experiment(I_grid=9, K=5, device=device)
    u = pde_model.sample_ground_truth(3)
    loss, x_hat = bilevel_model(u)
    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    params = bilevel_model.get_learned_params()
    assert "lambda" in params


# Test 6: Tikhonov regularizer
def test_tikhonov():
    from Regulariser.Tikhonov import TikhonovRegularizer
    reg = TikhonovRegularizer(lambda_init=1.0, device=device).to(device)

    # Test with images
    x = torch.rand(2, 1, 16, 16, device=device)
    energy = reg.g(x)
    grad = reg.grad(x)
    assert energy.shape == (2,), f"Energy shape: {energy.shape}"
    assert grad.shape == x.shape, f"Grad shape: {grad.shape}"

    # Test with custom L
    L = torch.eye(10, device=device)
    reg2 = TikhonovRegularizer(L=L, lambda_init=0.5, device=device).to(device)
    x2 = torch.rand(3, 10, device=device)
    energy2 = reg2.g(x2)
    assert energy2.shape == (3,)


# Test 7: calculate_levels
def test_calculate_levels():
    from train import calculate_levels

    L_sgd, N_sgd = calculate_levels(0.5, "SGD")
    assert L_sgd >= 1, f"L_sgd={L_sgd}"
    assert all(n >= 1 for n in N_sgd), f"N_sgd has zeros: {N_sgd}"

    L_gd, N_gd = calculate_levels(0.5, "GD")
    assert L_gd >= 1, f"L_gd={L_gd}"

    L_fista, N_fista = calculate_levels(0.5, "FISTA")
    assert L_fista >= 1, f"L_fista={L_fista}"
    assert all(n >= 1 for n in N_fista), f"N_fista has zeros: {N_fista}"


# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("  Multilevel Bilevel DFO Pipeline - Smoke Tests")
    print("=" * 60)

    print("\n[FISTA Solver]")
    test("FISTA converges on quadratic", test_fista_solver)
    test("reconstruct_fista with TV denoising", test_reconstruct_fista)

    print("\n[CT Physics]")
    test("DeepInverse Tomography forward/adjoint", test_ct_physics)

    print("\n[Phantom Generation]")
    test("Random triangle phantom", test_phantom_triangle)
    test("PhantomDataset", test_phantom_dataset)

    print("\n[CT Experiment]")
    test("CT experiment setup", test_ct_experiment_setup)
    test("CT evaluate_objective with A_dagger init", test_ct_evaluate_objective)
    test("CTSmoothedTV regularizer", test_ct_smoothed_tv)

    print("\n[PDE Forward Model]")
    test("2D Laplacian assembly", test_pde_laplacian)
    test("PDE forward model", test_pde_forward_model)
    test("PDE lower-level solve", test_pde_solve)
    test("PDE bilevel model", test_pde_bilevel)

    print("\n[Tikhonov Regularizer]")
    test("Tikhonov energy + gradient", test_tikhonov)

    print("\n[Level Schedule]")
    test("calculate_levels for SGD/GD/FISTA", test_calculate_levels)

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if FAIL > 0 else 0)
