import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    # TODO: выполнить обмены по кольцу с суммированием и вернуть chunks[rank].

    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    for step in range(1, world_size):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        recv_buf = torch.empty_like(chunks[recv_idx])

        req_send = dist.isend(chunks[send_idx], dst = right)
        req_recv = dist.irecv(recv_buf, src = left)

        req_send.wait()
        req_recv.wait()

        chunks[recv_idx] += recv_buf

    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    # TODO: передать шарды по кольцу и вернуть torch.cat(chunks).

    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    for step in range(1, world_size):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        recv_buf = torch.empty_like(shard)

        req_send = dist.isend(chunks[send_idx], dst = right)
        req_recv = dist.irecv(recv_buf, src = left)

        req_send.wait()
        req_recv.wait()

        chunks[recv_idx].copy_(recv_buf)

    return torch.cat(chunks)


def ring_all_reduce(tensor):
    # TODO: последовательно вызвать две функции выше. Здесь нужна сумма, не среднее.
    
    shard = ring_reduce_scatter(tensor)
    return ring_all_gather(shard)
