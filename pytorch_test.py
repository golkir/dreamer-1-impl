import torch
from torch import nn
from torch.distributions import Normal, MultivariateNormal

# if __name__ == "__main__":
#     x = torch.randn(2, 3)
#     x_transposed = x.t()  # Transposing changes metadata strides, making it non-contiguous

#     # This will crash!x
#     # y = x_transposed.view(6)
#     # RuntimeError: view size is not compatible with input tensor's size and stride...

#     """
#     two gaussians are created with different parameters
#     Batch shape is two
#     """
#     mean = torch.tensor([0., 10.])
#     std  = torch.tensor([1., 2.])

#     dist = Normal(mean, std)

#     print(dist.batch_shape, "Batch shape")
#     print(dist.event_shape, "Event shape")

#     cov = torch.diag(torch.tensor([1.0, 1.0]))

#     mv = MultivariateNormal(mean,cov)

#     print(mv.batch_shape, mv.event_shape)


def test_unflatten():
    print("here")
    input = torch.randn(10, 30)
    # With tuple of ints
    m = nn.Sequential(nn.Linear(30, 4), nn.Unflatten(1, (2, 2)))  # (10,4)
    output = m(input)
    # output.size() # (10,2,2)
    print(output.shape, "output shape")
    # With torch.Size
    # m = nn.Sequential(
    #     nn.Linear(50, 50),
    #     nn.Unflatten(1, torch.Size([2, 5, 5]))
    # )
    # output = m(input)
    # output.size()

    """
    how to fix
    x.contiguous().view(...): Forces PyTorch to copy the data into a brand new, sequential block of memory first, allowing .view() to work.
    x.reshape(...): A safer choice in modern PyTorch. .reshape() is smart: if the tensor is contiguous, it works exactly like .view() without copying data. If it isn't contiguous, it automatically clones the data under the hood so your code doesn't crash.
    """


if __name__ == "__main__":

    def process_orders(orders, discounts=[]):
        total = 0
        for order in orders:
            price = order["price"]
            if order["type"] == "vip":
                price = price * 0.8
            discounts.append(order["id"])
            total += price
        return total, discounts

    o = [{"id": 4, "price": 10,"type": "vip"}, {"id": 44, "price": 20,"type": "vip"}]

    total, discounts = process_orders(o)

    print(total, discounts)

    total1, discounts1 = process_orders(o)

    print(total1, discounts1)
    print(total, discounts)


