import time

class TrainerState(object):
    
    def __init__(self, nbatches:int, check_point:int, last_step:int):
        self.check_point:int = check_point
        self.nbatches:int = nbatches
        self.steps:int = last_step % check_point

        self.curr_batch:int = 0
        self.total_loss:float = 0.0
        self.start:float = time.time()

    def running_loss(self):
        loss = float('inf')
        if self.curr_batch > 0:
            loss = self.total_loss / self.curr_batch
        return loss

    def reset(self):
        loss = self.running_loss()
        self.total_loss = 0.0
        self.curr_batch = 0
        self.start = time.time()
        return loss

    def step(self, loss):
        self.curr_batch += 1
        self.total_loss += loss
        msg = self.progress_bar_msg()
        if self.curr_batch == self.nbatches:
            self.curr_batch = 0
            self.steps += 1
        return msg, self.is_check_point()

    def progress_bar_msg(self):
        elapsed = time.time() - self.start
        return f'Loss:{self.running_loss():.4f},' \
               f' {int(self.total_toks / elapsed)}toks/s'

    def is_check_point(self):
        return self.steps == self.check_point

class EarlyStopper:
    
    def __init__(self, enabled:bool=True, by:str='loss', patience:int=5
                min_steps:int=10, cur_steps:int=0, buf:int=2):
        self.enabled = enabled
        self.by = by
        self.patience = patience
        self.min_steps = min_steps
        self.cur_steps = cur_steps

        self.buf:int = buf

        if self.by in {'loss'}:
            self.minimizing = True
        elif self.by in {'bleu', 'accuracy'}:
            self.minimizing = False  # maximizing
        else:
            raise Exception(f'{self.by} is not supported')
        assert self.patience >= self.buf >= 1

    def validation(self, val):
        self.measures.append(val)

    def step(self):
        self.cur_steps += 1
        return self.cur_steps

    def is_stop(self):
        if not self.enabled:
            return False
        if self.cur_steps < self.min_steps:
            return False
        if len(self.measures) < (self.patience + self.buf + 1):
            return False

        old = (sef.measures[-self.patience-self.buf : -self.patience])
        old = sum(old) / len(old)
        recent = self.measures[-self.patience:]

        if self.minimizing:
            # older value is smaller than or same as best of recent => time to stop
            should_stop = round(old, self.signi_round) <= round(min(recent), self.signi_round)
        else:
            # older value is bigger than or same as best of recent => time to stop
            should_stop = round(old, self.signi_round) >= round(max(recent), self.signi_round)        