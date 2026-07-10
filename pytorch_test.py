import torch
from torch.distributions import Normal, MultivariateNormal
if __name__ == "__main__":
    x = torch.randn(2, 3)
    x_transposed = x.t()  # Transposing changes metadata strides, making it non-contiguous

    # This will crash!x
    # y = x_transposed.view(6)  
    # RuntimeError: view size is not compatible with input tensor's size and stride...

    """
    two gaussians are created with different parameters
    Batch shape is two
    """
    mean = torch.tensor([0., 10.])
    std  = torch.tensor([1., 2.])
    
    dist = Normal(mean, std)

    print(dist.batch_shape, "Batch shape")
    print(dist.event_shape, "Event shape")

    cov = torch.diag(torch.tensor([1.0, 1.0]))

    mv = MultivariateNormal(mean,cov)

    print(mv.batch_shape, mv.event_shape)

    




    """
    how to fix
    x.contiguous().view(...): Forces PyTorch to copy the data into a brand new, sequential block of memory first, allowing .view() to work.
    x.reshape(...): A safer choice in modern PyTorch. .reshape() is smart: if the tensor is contiguous, it works exactly like .view() without copying data. If it isn't contiguous, it automatically clones the data under the hood so your code doesn't crash.
    """

    