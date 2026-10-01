import multiprocessing as mp
import time

import torch
from nixl._api import nixl_agent


def worker(device, pipe):
    torch.cuda.set_device(device)
    tensor = (
        torch.arange(32, dtype=torch.float32, device=f"cuda:{device}")
        if device == 0
        else torch.zeros(32, device=f"cuda:{device}")
    )
    torch.cuda.synchronize()
    name = f"ouro-nixl-{device}"
    agent = nixl_agent(name)
    registration = agent.register_memory(tensor)
    pipe.send(
        dict(name=name, metadata=agent.get_agent_metadata(), ptr=tensor.data_ptr(), device=device)
    )
    peer = pipe.recv()
    agent.add_remote_agent(peer["metadata"])
    pipe.send("ready")
    pipe.recv()
    if device == 0:
        handle = agent.initialize_xfer(
            "WRITE",
            agent.get_xfer_descs(tensor),
            agent.get_xfer_descs([(peer["ptr"], 128, peer["device"])], "VRAM"),
            peer["name"],
            notif_msg=b"done",
            backends=["UCX"],
        )
        agent.transfer(handle)
        while agent.check_xfer_state(handle) != "DONE":
            time.sleep(0.001)
        agent.release_xfer_handle(handle)
    else:
        while not agent.get_new_notifs():
            time.sleep(0.001)
    pipe.send(tensor.cpu().tolist())
    pipe.recv()
    agent.remove_remote_agent(peer["name"])
    agent.deregister_memory(registration)
    pipe.close()


if __name__ == "__main__":
    context = mp.get_context("spawn")
    peers = []
    for device in range(2):
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(device, child))
        process.start()
        child.close()
        peers.append((process, parent))
    infos = [p.recv() for _, p in peers]
    for i, (_, p) in enumerate(peers):
        p.send(infos[1 - i])
    assert all(p.recv() == "ready" for _, p in peers)
    for _, p in peers:
        p.send("go")
    values = [p.recv() for _, p in peers]
    print("VALUES", values, flush=True)
    for _, p in peers:
        p.send("stop")
    for process, p in peers:
        process.join()
        p.close()
        assert process.exitcode == 0
    assert values[0] == values[1]
    print("MULTIPROCESS_NIXL_PASSED", flush=True)
