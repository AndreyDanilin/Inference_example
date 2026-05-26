import pytest

from inference_lab.kernels.benchmarks import available_cuda, run_kernel_correctness


@pytest.mark.cuda
def test_triton_kernels_match_torch_when_cuda_is_available() -> None:
    if not available_cuda():
        pytest.skip("CUDA and Triton are required for kernel correctness checks")

    results = run_kernel_correctness(kernels=["matmul", "rmsnorm"])

    assert {result.kernel for result in results} == {"matmul", "rmsnorm"}
    assert all(result.passed for result in results)
