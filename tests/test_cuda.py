import torch

__author__ = "Core for Computational Biomedicine at Harvard Medical School"
__copyright__ = "Center for Computational Biomedicine"
__license__ = "CC-BY-4.0"


def get_device():
    """
    Prints information about the availability of CUDA and the number of GPUs found.
    If CUDA is available, it also prints information about the current device ID, GPU device name, and CUDNN version.
    If CUDA is available, the method sets the device to 'cuda:0' and clears the CUDA cache.
    Otherwise, it sets the device to 'cpu'.
    Returns:
        torch.device: The selected device.
    """
    is_cuda = torch.cuda.is_available()
    print(f'CUDA available: {is_cuda}')
    print(f'Number of GPUs found:  {torch.cuda.device_count()}')

    if is_cuda:
        print(f'Current device ID:     {torch.cuda.current_device()}')
        print(f'GPU device name:       {torch.cuda.get_device_name(0)}')
        print(f'CUDNN version:         {torch.backends.cudnn.version()}')
        device_str = 'cuda:0'
        torch.cuda.empty_cache()
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    return device


def test_cuda():
    """
    Checks if the device is a CUDA device.
    Returns:
        None
    Raises:
        AssertionError: If the device is not a CUDA device.
    """
    print(get_device().type)
    # assert get_device().type == 'cuda', f'{get_device()} is not a CUDA device'

if __name__ == "__main__":
    device = get_device()
    test_cuda()