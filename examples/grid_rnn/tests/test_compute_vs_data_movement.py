from random import randint
from time import time

import torch


def create_data_pool(tensor_size, pool_size, device, continuous):
    return None if continuous else [
        torch.randn(tensor_size, device=device) for _ in range(pool_size)
    ]


def gen_slices(hidden_size, batch_size, device, data_pool):
    if data_pool is None:
        return [
            torch.randn(1, hidden_size, device=device)
            for _ in range(batch_size)
        ]

    row_size = data_pool[0].size()[0]
    pool_size = len(data_pool)
    return [
        data_pool[randint(0, pool_size - 1)][randint(0, row_size - 1), :].view(
            1, hidden_size) for _ in range(batch_size)
    ]


def __make_test(
        hidden_size,
        batch_size,
        input_slices,
        i,  # index to reference output list
        stack_time,
        reshape_time,
        lstm_time,
        unbind_time,
        view_time,
        device,
        repeat=1):
    lstm = torch.nn.LSTMCell(
        input_size=hidden_size, hidden_size=hidden_size, bias=True).to(device)
    for _ in range(repeat):
        start = time()
        x = torch.stack(input_slices, dim=0)
        stack_time[i] += (time() - start)

        start = time()
        x = torch.reshape(x, [batch_size, hidden_size])
        reshape_time[i] += (time() - start)

        start = time()
        lstm(x)
        lstm_time[i] += (time() - start)

        start = time()
        xs = torch.unbind(x, dim=0)
        unbind_time[i] += (time() - start)

        start = time()
        xs = [x.view(1, -1) for x in xs]
        view_time[i] += (time() - start)


def test_compute_vs_data_movement(batch_size):
    ROW_SIZE = 128
    POOL_SIZE = 2048

    repeat = 20
    hidden_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for continuous in [
            True,
            False,
    ]:
        for device in [
                "cpu",
                "cuda",
        ]:
            stack_time = [0.] * len(hidden_sizes)
            reshape_time = [0.] * len(hidden_sizes)
            lstm_time = [0.] * len(hidden_sizes)
            unbind_time = [0.] * len(hidden_sizes)
            view_time = [0.] * len(hidden_sizes)

            for i, hidden_size in enumerate(hidden_sizes):
                data = create_data_pool((ROW_SIZE, hidden_size), POOL_SIZE,
                                        device, continuous)
                xs = gen_slices(hidden_size, batch_size, device, data)

                __make_test(hidden_size, batch_size, xs, i, stack_time,
                            reshape_time, lstm_time, unbind_time, view_time,
                            device, repeat)

            # average and "s" to "ms"
            stack_time = [x / repeat * 1000. for x in stack_time]
            reshape_time = [x / repeat * 1000. for x in reshape_time]
            lstm_time = [x / repeat * 1000. for x in lstm_time]
            unbind_time = [x / repeat * 1000. for x in unbind_time]
            view_time = [x / repeat * 1000. for x in view_time]

            out_info = [
                ["stack", stack_time],
                ["reshape", reshape_time],
                ["lstm", lstm_time],
                ["unbind", unbind_time],
                ["view", view_time],
            ]

            for name, info in out_info:
                prefix = f"{name}_{device}" + ("" if continuous else "_dis")
                print("%s = [%s];" % (prefix, ",".join(
                    map("{:.6f}".format, info))))


if __name__ == "__main__":
    for bs in [
            16,
            32,
            64,
            128,
            256,
            512,
    ]:
        print("\n%%%%\nbatch_size = %d;" % (bs))  # for matlab codes.
        test_compute_vs_data_movement(bs)
