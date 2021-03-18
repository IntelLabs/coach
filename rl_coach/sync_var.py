class SyncVar:
    # 全局同步变量
    def __init__(self):
        # for architecture/wait_for_all_workers_barrier
        self.lock_counter = None
        self.release_counter = None
        # for agent/train_network
        self.min_batch_size = None
        self.agent_lock_counter = None
        self.agent_release_counter = None


global global_sync_obj
